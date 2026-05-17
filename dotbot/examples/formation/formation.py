"""
formation.py
============
Move active DotBots into the shape of a single letter (A-Z).

Usage:
    python formation.py --letter A
    python formation.py --letter B --cell-size 200

The script:
  1. Queries the controller for the arena size and the active DotBots.
  2. Renders the letter as a 5x7 bitmap centered in the arena.
  3. Uses the Hungarian algorithm to match each bot to the nearest letter pixel.
  4. Sends one waypoint per bot. If there are more bots than pixels, the extras
     park in a row along the bottom edge of the arena.

LED conventions while the script runs:
  - red   : moving
  - green : arrived at letter pixel
  - blue  : parked (extra robot, not part of the letter)

Requirements:
    pip install websockets httpx numpy scipy rich

The dotbot-controller must already be running.
"""

import asyncio
import math
from dataclasses import dataclass

import click
import numpy as np
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment

from dotbot.examples.formation.font5x7 import FONT_5X7, GRID_H, GRID_W
from dotbot.models import (
    DotBotLH2Position,
    DotBotModel,
    DotBotQueryModel,
    DotBotRgbLedCommandModel,
    DotBotStatus,
    DotBotWaypoints,
    WSRgbLed,
    WSWaypoints,
)
from dotbot.protocol import ApplicationType, ControlModeType
from dotbot.rest import RestClient, rest_client
from dotbot.websocket import DotBotWsClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APPLICATION = ApplicationType.DotBot

CELL_SIZE_DEFAULT = 150   # mm between adjacent letter pixels (DotBot diameter ~120 mm)
THRESHOLD_DEFAULT = 60    # mm — arrival threshold (one bot radius; firmware default is 100)
SEND_RETRY = 0.1          # s — interval between waypoint resends while waiting for AUTO
ARRIVAL_POLL = 0.5        # s — polling interval while waiting for bots to arrive
PARK_Y_FRAC = 0.05        # vertical position of parking row (fraction of arena height)
PARK_MARGIN_FRAC = 0.05   # horizontal margin of parking row (fraction of arena width)
DEFAULT_TIMEOUT = 60.0    # s — max wait for the formation to settle

LED_MOVING = (255, 0, 0)    # red
LED_ARRIVED = (0, 255, 0)   # green
LED_PARKED = (0, 0, 255)    # blue
LED_OFF = (0, 0, 0)


# ---------------------------------------------------------------------------
# Letter geometry
# ---------------------------------------------------------------------------


def letter_pixels(letter: str) -> list[tuple[int, int]]:
    """Return ON pixel grid coordinates (col, row) for the given letter."""
    letter = letter.upper()
    if letter not in FONT_5X7:
        raise ValueError(
            f"Unsupported letter: {letter!r}. Supported: {sorted(FONT_5X7)}"
        )
    bitmap = FONT_5X7[letter]
    return [
        (col, row)
        for row, line in enumerate(bitmap)
        for col, ch in enumerate(line)
        if ch != " "
    ]


def render_letter_targets(
    letter: str, arena_w: int, arena_h: int, cell_size: float
) -> list[tuple[float, float]]:
    """Convert a letter into a list of (x, y) arena coordinates in mm, centered."""
    pixels = letter_pixels(letter)
    bbox_w = (GRID_W - 1) * cell_size
    bbox_h = (GRID_H - 1) * cell_size
    if bbox_w > arena_w or bbox_h > arena_h:
        raise ValueError(
            f"Letter bounding box {bbox_w:.0f}x{bbox_h:.0f} mm does not fit in "
            f"arena {arena_w}x{arena_h} mm. Reduce --cell-size."
        )
    cx, cy = arena_w / 2, arena_h / 2
    x0 = cx - bbox_w / 2
    y_top = cy - bbox_h / 2
    # Arena uses image-style coordinates: y=0 is the top of the dashboard,
    # y=arena_h is the bottom. Bitmap row 0 is also the top of the glyph,
    # so map row directly to y without flipping.
    return [
        (x0 + col * cell_size, y_top + row * cell_size)
        for col, row in pixels
    ]


def park_positions(
    n_extras: int, arena_w: int, arena_h: int
) -> list[tuple[float, float]]:
    """Evenly space extras along the bottom edge of the arena (high y in image coords)."""
    if n_extras == 0:
        return []
    y = arena_h * (1 - PARK_Y_FRAC)
    margin = arena_w * PARK_MARGIN_FRAC
    if n_extras == 1:
        return [(arena_w / 2, y)]
    span = arena_w - 2 * margin
    return [(margin + i * span / (n_extras - 1), y) for i in range(n_extras)]


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssignmentResult:
    formation: dict[str, tuple[float, float]]  # address -> letter-pixel target
    extras: list[DotBotModel]                  # bots that didn't get a letter pixel


def assign_bots(
    bots: list[DotBotModel], targets: list[tuple[float, float]]
) -> AssignmentResult:
    """Hungarian assignment of bots to letter targets, minimising total travel."""
    n_bots, n_targets = len(bots), len(targets)
    cost = np.zeros((n_bots, n_targets))
    for i, b in enumerate(bots):
        for j, (tx, ty) in enumerate(targets):
            cost[i, j] = math.hypot(b.lh2_position.x - tx, b.lh2_position.y - ty)
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_rows = {int(i) for i in row_ind}
    formation = {bots[int(i)].address: targets[int(j)] for i, j in zip(row_ind, col_ind)}
    extras = [b for idx, b in enumerate(bots) if idx not in assigned_rows]
    return AssignmentResult(formation=formation, extras=extras)


# ---------------------------------------------------------------------------
# Bot commands
# ---------------------------------------------------------------------------


async def set_led(
    ws: DotBotWsClient, address: str, rgb: tuple[int, int, int]
) -> None:
    r, g, b = rgb
    await ws.send(
        WSRgbLed(
            cmd="rgb_led",
            address=address,
            application=APPLICATION,
            data=DotBotRgbLedCommandModel(red=r, green=g, blue=b),
        )
    )


async def confirm_all_in_auto(
    ws: DotBotWsClient,
    client: RestClient,
    targets: dict[str, tuple[float, float]],
    threshold: int,
) -> None:
    """Send each bot a single waypoint and re-send until every one confirms AUTO mode.

    Uses one batched REST poll per cycle (not one per bot) so the controller
    sees ~1 req/100ms instead of N req/100ms — keeps it responsive when N is large.
    """
    msgs = {
        addr: WSWaypoints(
            cmd="waypoints",
            address=addr,
            application=APPLICATION,
            data=DotBotWaypoints(
                threshold=threshold,
                waypoints=[DotBotLH2Position(x=t[0], y=t[1])],
            ),
        )
        for addr, t in targets.items()
    }
    pending = set(targets)
    while pending:
        for addr in pending:
            await ws.send(msgs[addr])
        await asyncio.sleep(SEND_RETRY)
        bots = await client.fetch_dotbots()
        in_auto = {b.address for b in bots if b.mode == ControlModeType.AUTO}
        pending -= in_auto


async def wait_for_arrival(
    client: RestClient,
    ws: DotBotWsClient,
    formation: dict[str, tuple[float, float]],
    threshold: int,
    timeout: float,
) -> set[str]:
    """Poll bot positions; flip LED to green per bot on arrival."""
    arrived: set[str] = set()
    deadline = asyncio.get_event_loop().time() + timeout
    total = len(formation)
    while len(arrived) < total:
        if asyncio.get_event_loop().time() > deadline:
            rprint(
                f"  [yellow]Timeout: {len(arrived)}/{total} arrived[/yellow]"
            )
            break
        await asyncio.sleep(ARRIVAL_POLL)
        bots = await client.fetch_dotbots()
        for b in bots:
            if b.address in arrived or b.address not in formation:
                continue
            tx, ty = formation[b.address]
            if math.hypot(b.lh2_position.x - tx, b.lh2_position.y - ty) <= threshold:
                arrived.add(b.address)
                await set_led(ws, b.address, LED_ARRIVED)
                rprint(
                    f"  [green]✓[/green] {b.address} arrived "
                    f"([bold]{len(arrived)}/{total}[/bold])"
                )
    return arrived


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _display_plan(
    console: Console,
    letter: str,
    formation: dict[str, tuple[float, float]],
    parks: dict[str, tuple[float, float]],
) -> None:
    table = Table(
        title=f"Plan: form letter '{letter.upper()}'",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("address", style="dim")
    table.add_column("role")
    table.add_column("target x (mm)", justify="right")
    table.add_column("target y (mm)", justify="right")
    for addr, (tx, ty) in formation.items():
        table.add_row(addr, "letter", f"{tx:.0f}", f"{ty:.0f}")
    for addr, (tx, ty) in parks.items():
        table.add_row(addr, "park", f"{tx:.0f}", f"{ty:.0f}")
    console.print(table)


async def run(
    host: str,
    port: int,
    use_https: bool,
    letter: str,
    cell_size: float,
    threshold: int,
    timeout: float,
) -> None:
    console = Console()

    async with rest_client(host, port, use_https) as client:
        map_size = await client.fetch_map_size()
        bots = await client.fetch_dotbots(
            query=DotBotQueryModel(status=DotBotStatus.ACTIVE)
        )
        arena_w, arena_h = map_size.width, map_size.height

        rprint(f"[cyan]Arena[/cyan]: {arena_w} x {arena_h} mm")
        rprint(f"[cyan]Active DotBots[/cyan]: {len(bots)}")

        targets = render_letter_targets(letter, arena_w, arena_h, cell_size)
        rprint(
            f"[cyan]Letter[/cyan]: '{letter.upper()}' → "
            f"[bold]{len(targets)}[/bold] pixels at {cell_size:.0f} mm spacing"
        )

        if len(bots) < len(targets):
            rprint(
                f"[bold red]ERROR:[/bold red] Need [bold]{len(targets)}[/bold] "
                f"bots for '{letter.upper()}', have [bold]{len(bots)}[/bold]."
            )
            return

        assignment = assign_bots(bots, targets)
        parks = park_positions(len(assignment.extras), arena_w, arena_h)
        park_targets = {b.address: parks[i] for i, b in enumerate(assignment.extras)}

        _display_plan(console, letter, assignment.formation, park_targets)

        ws = DotBotWsClient(host, port)
        await ws.connect()
        try:
            # Step 1: everyone red (about to move).
            await asyncio.gather(*[set_led(ws, b.address, LED_MOVING) for b in bots])

            # Step 2: send every bot its target; retry until each confirms AUTO mode.
            all_targets = {**assignment.formation, **park_targets}
            await confirm_all_in_auto(ws, client, all_targets, threshold)
            rprint(
                "[green]All bots in AUTO mode[/green] — "
                "waiting for letter-pixel arrival ..."
            )

            # Step 3: mark parked bots blue (they have a separate, non-letter goal).
            await asyncio.gather(*[
                set_led(ws, addr, LED_PARKED) for addr in park_targets
            ])

            # Step 4: poll until each letter-pixel bot is within threshold.
            await wait_for_arrival(
                client, ws, assignment.formation, threshold, timeout
            )

            rprint(f"[bold green]Formation '{letter.upper()}' complete.[/bold green]")
        finally:
            await ws.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--host", default="localhost", show_default=True, help="Controller host.")
@click.option("--port", type=int, default=8000, show_default=True, help="Controller port.")
@click.option(
    "--https/--no-https",
    default=False,
    help="Use HTTPS for the REST client.",
)
@click.option(
    "-l",
    "--letter",
    required=True,
    help="Single letter (A-Z) to form.",
)
@click.option(
    "--cell-size",
    type=float,
    default=CELL_SIZE_DEFAULT,
    show_default=True,
    help="Distance between adjacent letter pixels in mm.",
)
@click.option(
    "--threshold",
    type=int,
    default=THRESHOLD_DEFAULT,
    show_default=True,
    help="Arrival threshold in mm.",
)
@click.option(
    "--timeout",
    type=float,
    default=DEFAULT_TIMEOUT,
    show_default=True,
    help="Max seconds to wait for the formation to complete.",
)
def main(host, port, https, letter, cell_size, threshold, timeout) -> None:
    """Move active DotBots into the shape of one letter."""
    if len(letter) != 1:
        raise click.BadParameter("--letter must be a single character.")
    asyncio.run(run(host, port, https, letter, cell_size, threshold, timeout))


if __name__ == "__main__":
    main()
