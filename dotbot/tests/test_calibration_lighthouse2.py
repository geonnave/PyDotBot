"""Tests for the LH2 calibration solve, the schema 2 file and the identity.

Synthetic captures are built by inverting one chosen station matrix, so the
correspondences are exactly consistent and the solver's own error is the only
thing under test. A raw count is an integer, and one count unit is worth
about 0.1 mm on the floor, so the declared coordinates are derived from the
integer counts rather than the other way round.
"""

import math
import tomllib

import numpy as np
import pytest

from dotbot.calibration import lighthouse2
from dotbot.calibration.lighthouse2 import (
    LH_PERIODS,
    Frame,
    LH2Counts,
    LighthouseManager,
    Placement,
    Sample,
    apply_homography,
    calculate_camera_point,
    read_calibration_file,
    render_calibration,
    resolve_calibration_path,
)
from dotbot.calibration.points import resolve_points
from dotbot.calibration.wire import calibration_payload, unpack_payload
from dotbot.bounds import Bounds, BoundsRegistry

# A plausible wall-mounted station: the magnitude of perspective row real
# calibration files carry.
H_TRUE = np.array(
    [
        [1523.4, -38.2, 1012.7],
        [41.9, 1531.8, 988.3],
        [0.2134, -0.0871, 1.0],
    ],
    dtype=np.float64,
)

ARENA = Bounds(0, 0, 2000, 2000, "arena", ("top", "left"))


def counts_for_camera_point(cam_x, cam_y, lh_index=0) -> LH2Counts:
    """Inverse of `calculate_camera_point`, for building synthetic captures."""
    period = LH_PERIODS[lh_index]
    half_sum = math.atan(-cam_x)
    half_diff = math.pi / 3 + math.asin(-cam_y * math.tan(math.pi / 6))
    a1, a2 = half_sum - half_diff, half_sum + half_diff
    while a1 < 0:
        a1 += math.pi
        a2 += math.pi
    scale = period / 8 / (2 * math.pi)
    return LH2Counts(lh_index, a1 * scale, a2 * scale)


def _floor_from_camera(homography, cam_x, cam_y):
    return apply_homography(homography, np.array([[cam_x, cam_y]]))[0]


def _integer_counts(cam_x, cam_y, lh_index=0) -> tuple[int, int]:
    counts = counts_for_camera_point(cam_x, cam_y, lh_index)
    return (round(counts.count1), round(counts.count2))


def _sample(station, point, count1, count2, reads=1) -> Sample:
    return Sample(
        station=station,
        point=point,
        count1=[count1] * reads,
        count2=[count2] * reads,
    )


def _grid_camera_points(n_side):
    """Camera points spread over the part of the view the arena occupies."""
    return [
        (-0.30 + 0.60 * i / (n_side - 1), -0.30 + 0.60 * j / (n_side - 1))
        for i in range(n_side)
        for j in range(n_side)
    ]


def _consistent_placement(camera_points, station=0, reads=1, jitter_mm=0.0, seed=0):
    """A placement whose declared points are `H_TRUE`'s image of the counts.

    With `jitter_mm` the declared coordinates are displaced by a Gaussian per
    axis, which is what a hand-placed photodiode does.
    """
    rng = np.random.default_rng(seed)
    points, samples = [], []
    for index, (cam_x, cam_y) in enumerate(camera_points):
        count1, count2 = _integer_counts(cam_x, cam_y, station)
        back = calculate_camera_point(LH2Counts(station, count1, count2))
        floor = _floor_from_camera(H_TRUE, back[0], back[1])
        if jitter_mm:
            floor = floor + rng.normal(0.0, jitter_mm, 2)
        points.append((float(floor[0]), float(floor[1])))
        samples.append(_sample(station, index, count1, count2, reads))
    return Placement(index=0, at="synthetic", points_mm=points, samples=samples)


# --- the camera model -------------------------------------------------------


def test_camera_points():
    counts = LH2Counts(lh_index=1, count1=49341, count2=85887)
    x, y = calculate_camera_point(counts)
    assert x == pytest.approx(-0.43435315273542)
    assert y == pytest.approx(0.1512338330873567)


# --- the solver -------------------------------------------------------------


def test_four_points_solve_exactly_and_place_an_off_centre_point():
    """Four noiseless points fix the map, and it holds 900 mm off centre."""
    corners = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
    placement = _consistent_placement(corners)
    manager = LighthouseManager(placements=[placement])
    station = manager.solve()[0]

    assert station.points == 4
    # A four-point fit reprojects its own points exactly; the residual is the
    # solver's own arithmetic, tens of nanometres on a matrix of this scale.
    assert station.residual_mm < 1e-3

    centre = np.mean(np.array(placement.points_mm), axis=0)
    # A camera point whose true floor position is about 900 mm off the figure
    # centre, so the check is extrapolation rather than interpolation.
    off_x, off_y = 0.55, 0.55
    count1, count2 = _integer_counts(off_x, off_y)
    back = calculate_camera_point(LH2Counts(0, count1, count2))
    truth = _floor_from_camera(H_TRUE, back[0], back[1])
    solved = _floor_from_camera(station.matrix, back[0], back[1])

    assert np.linalg.norm(truth - centre) == pytest.approx(900, abs=250)
    assert np.linalg.norm(solved - truth) < 0.01


def test_sixteen_noisy_points_solve_by_least_squares_with_a_residual():
    """Over-determined and noisy: a residual, and no exception."""
    placement = _consistent_placement(
        _grid_camera_points(4), reads=25, jitter_mm=3.0, seed=7
    )
    manager = LighthouseManager(placements=[placement])
    station = manager.solve()[0]

    assert station.points == 16
    # Four points fix eight unknowns, so sixteen points leave the residual at
    # roughly the per-point placement sigma. It is never zero with noise.
    assert 0.5 < station.residual_mm < 5.0


def test_a_solve_needs_four_points():
    placement = _consistent_placement([(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2)])
    manager = LighthouseManager(placements=[placement])
    with pytest.raises(ValueError, match="4 points a homography needs"):
        manager.solve()


def test_two_stations_are_solved_from_the_same_placement():
    """A placement seen by two stations ties both into the same frame."""
    corners = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
    placement = _consistent_placement(corners)
    for index, (cam_x, cam_y) in enumerate(corners):
        count1, count2 = _integer_counts(cam_x, cam_y, 1)
        placement.samples.append(_sample(1, index, count1, count2))

    stations = LighthouseManager(placements=[placement]).solve()
    assert [s.index for s in stations] == [0, 1]
    assert all(s.points == 4 for s in stations)


def test_reads_are_averaged_before_the_solve():
    sample = _sample(0, 0, 100, 200, reads=1)
    sample.count1 = [100, 102, 104]
    sample.count2 = [200, 200, 200]
    assert sample.reads == 3
    assert sample.mean_counts().count1 == pytest.approx(102.0)


# --- the file ---------------------------------------------------------------


def _saved(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setattr(lighthouse2, "CALIBRATION_DIR", tmp_path)
    corners = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
    placement = _consistent_placement(corners, reads=3)
    manager = LighthouseManager(placements=[placement], **kwargs)
    manager.solve()
    return manager, manager.save_calibration(tag=kwargs.pop("tag", None))


def test_save_writes_schema_2_into_the_frame_directory(monkeypatch, tmp_path):
    _, path = _saved(monkeypatch, tmp_path)

    assert path.parent == tmp_path / "calibrations" / "inria-aio-c"
    parsed = tomllib.loads(path.read_text())
    assert parsed["schema_version"] == 2
    assert parsed["frame"]["name"] == "inria-aio-c"
    assert parsed["frame"]["false_origin_mm"] == [0, 0]
    assert parsed["frame"]["false_origin_at"]
    assert parsed["validity"]["valid_mm"] == [0, 0, 4000, 4500]
    assert parsed["metadata"]["robot"] == "dotbot-v3"
    assert len(parsed["metadata"]["id"]) == 16
    assert path.name.endswith(f"-{parsed['metadata']['id'][:8]}.toml")
    assert len(parsed["placement"]) == 1
    assert len(parsed["placement"][0]["samples"]) == 4
    assert parsed["placement"][0]["samples"][0]["count1"] == pytest.approx(
        parsed["placement"][0]["samples"][0]["count1"]
    )
    assert len(parsed["station"]) == 1
    assert parsed["station"][0]["solved_from"] == "direct"
    assert "calibration_distance_mm" not in parsed["metadata"]
    assert "num_lh_stations" not in parsed["metadata"]
    assert "calibration" not in parsed


def test_save_writes_no_legacy_out_sidecar(monkeypatch, tmp_path):
    _saved(monkeypatch, tmp_path)
    assert not (tmp_path / "calibration.out").exists()
    assert list(tmp_path.rglob("*.out")) == []


def test_schema_2_round_trips_and_re_solves_to_the_same_matrices_and_id(
    monkeypatch, tmp_path
):
    manager, path = _saved(monkeypatch, tmp_path)
    loaded = read_calibration_file(path)

    assert loaded.stored_id == loaded.id
    assert loaded.placements[0].points_mm == manager.placements[0].points_mm

    re_solved = LighthouseManager(
        placements=loaded.placements, frame=loaded.frame, valid_mm=loaded.valid_mm
    )
    re_solved.solve()
    assert re_solved.stations[0].homography == loaded.stations[0].homography
    assert re_solved.stations[0].residual_mm == loaded.stations[0].residual_mm

    written_again = render_calibration(loaded)
    assert f'id = "{loaded.id}"' in written_again


def test_schema_1_file_is_rejected(tmp_path):
    path = tmp_path / "calibration-2026-01-01T00-00-00Z-deadbeef.toml"
    path.write_text(
        'schema_version = 1\n[calibration]\ndata_hex = "00"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version 1"):
        read_calibration_file(path)


def test_calibration_id_ignores_the_descriptive_fields(monkeypatch, tmp_path):
    _, path = _saved(monkeypatch, tmp_path)
    original = read_calibration_file(path)
    before = original.id

    original.frame.false_origin_at = "somewhere else entirely, 2027"
    original.created_at = "2030-12-31T23:59:59Z"
    original.tag = "another-session"
    original.robot = "dotbot-v9"
    original.placements[0].at = "typed by hand"
    assert original.id == before


def test_calibration_id_moves_when_the_false_origin_moves(monkeypatch, tmp_path):
    _, path = _saved(monkeypatch, tmp_path)
    calibration = read_calibration_file(path)
    before = calibration.id

    calibration.frame.false_origin_mm = (1, 0)
    assert calibration.id != before


def test_calibration_id_moves_when_a_point_or_a_matrix_moves(monkeypatch, tmp_path):
    _, path = _saved(monkeypatch, tmp_path)

    calibration = read_calibration_file(path)
    before = calibration.id
    x, y = calibration.placements[0].points_mm[0]
    calibration.placements[0].points_mm[0] = (x + 1.0, y)
    assert calibration.id != before

    calibration = read_calibration_file(path)
    calibration.stations[0].homography[2][2] = 1.001
    assert calibration.id != before

    calibration = read_calibration_file(path)
    calibration.valid_mm = (0, 0, 5000, 5000)
    assert calibration.id != before


def test_resolve_by_id_prefix_under_the_frame_directory(monkeypatch, tmp_path):
    _, path = _saved(monkeypatch, tmp_path)
    calibration = read_calibration_file(path)

    resolved = resolve_calibration_path(
        calibration.id8, root=tmp_path / "calibrations"
    )
    assert resolved == path

    with pytest.raises(ValueError, match="no calibration matches"):
        resolve_calibration_path("ffffffff", root=tmp_path / "calibrations")


def test_resolve_prefers_an_actual_path(monkeypatch, tmp_path):
    _, path = _saved(monkeypatch, tmp_path)
    assert resolve_calibration_path(str(path)) == path


def test_wire_payload_is_float32_and_round_trips(monkeypatch, tmp_path):
    _, path = _saved(monkeypatch, tmp_path)
    calibration = read_calibration_file(path)

    payload = calibration_payload(calibration.stations)
    assert len(payload) == 1 + 36
    assert payload[0] == 1

    unpacked = unpack_payload(payload)[0]
    assert np.allclose(unpacked, calibration.stations[0].homography, rtol=1e-6)


# --- points and bounds -----------------------------------------------------


def test_arena_corner_marks_come_from_the_robot_geometry():
    """Two walls: the mark insets by the photodiode's clearance to each."""
    assert resolve_points("arena:corners", BoundsRegistry()) == [
        (50.0, 20.0),
        (2000.0, 20.0),
        (50.0, 2000.0),
        (2000.0, 2000.0),
    ]


def test_one_wall_insets_only_that_edge():
    registry = BoundsRegistry(named={"strip": Bounds(0, 0, 1000, 1000, "strip", ("top",))})
    assert resolve_points("strip:corners", registry) == [
        (0.0, 20.0),
        (1000.0, 20.0),
        (0.0, 1000.0),
        (1000.0, 1000.0),
    ]


def test_no_wall_resolves_to_the_exact_frame_corners():
    registry = BoundsRegistry(named={"open": Bounds(500, 700, 1000, 1000, "open")})
    assert resolve_points("open:corners", registry) == [
        (500.0, 700.0),
        (1500.0, 700.0),
        (500.0, 1700.0),
        (1500.0, 1700.0),
    ]


def test_a_bottom_wall_insets_by_the_rear_clearance():
    registry = BoundsRegistry(
        named={"back": Bounds(0, 0, 1000, 1000, "back", ("bottom", "right"))}
    )
    assert resolve_points("back:bottom-right", registry) == [(950.0, 920.0)]


def test_points_forms():
    registry = BoundsRegistry()
    assert resolve_points("1500,2500", registry) == [(1500.0, 2500.0)]
    assert resolve_points("arena", registry) == [(1000.0, 1000.0)]
    assert resolve_points("arena:top-right", registry) == [(2000.0, 20.0)]
    with pytest.raises(ValueError, match="unknown bounds"):
        resolve_points("nowhere", registry)


def test_bounds_resolution_forms():
    registry = BoundsRegistry()
    assert registry.resolve("annex").as_dict() == {
        "x": 0,
        "y": 2000,
        "w": 2000,
        "h": 2000,
    }
    assert registry.resolve("0,0,500,600").as_dict() == {
        "x": 0,
        "y": 0,
        "w": 500,
        "h": 600,
    }
    composite = registry.resolve("arena+wing")
    assert composite.as_dict() == {"x": 0, "y": 0, "w": 3330, "h": 4000}
    # The union keeps only the walls that lie on its own outline.
    assert set(composite.walls) == {"top", "left"}


def test_frame_defaults_name_the_anchor():
    frame = Frame()
    assert frame.name == "inria-aio-c"
    assert frame.false_origin_mm == (0, 0)
    assert "C405" in frame.false_origin_at


def test_slug_tag_rules():
    assert lighthouse2._slug_tag("office-2x2m") == "office-2x2m"
    assert lighthouse2._slug_tag("  a  b  ") == "a-b"
    assert lighthouse2._slug_tag("a/b\\c:d") == "a-b-c-d"
    assert lighthouse2._slug_tag("--keep_me.v2--") == "keep_me.v2"
    assert lighthouse2._slug_tag("..") == ""
    assert lighthouse2._slug_tag("***") == ""


# The same schema 2 fixture swarmit's test_helpers.py carries, so the two
# packers cannot drift.
FIXTURE_TOML = """\
schema_version = 2

[metadata]
created_at = "2026-09-10T09:12:00Z"
id = "3f9a1c07e2b845d6"
robot = "dotbot-v3"

[frame]
name = "inria-aio-c"
false_origin_mm = [0, 0]
false_origin_at = "arena top-left corner, against the door wall of C405"

[validity]
valid_mm = [0, 0, 4000, 4500]

[[placement]]
index = 0
at = "arena:corners"
points_mm = [[50.0, 20.0], [2000.0, 20.0], [50.0, 2000.0], [2000.0, 2000.0]]
captured_at = "2026-09-10T09:10:41Z"
samples = [
  { station = 0, point = 0, count1 = [41290], count2 = [51728] },
]

[[station]]
index = 0
solved_from = "direct"
points = 4
residual_mm = 0.0
homography = [[1523.4, -38.2, 1012.7], [41.9, 1531.8, 988.3], [0.2134, -0.0871, 1.0]]
"""


def test_the_wire_payload_is_pinned_against_the_shared_fixture(tmp_path):
    """Both repos build the same bytes from the same file."""
    import struct

    path = tmp_path / "calibration.toml"
    path.write_text(FIXTURE_TOML, encoding="utf-8")
    calibration = read_calibration_file(path)

    expected = bytes([1]) + struct.pack(
        "<9f", 1523.4, -38.2, 1012.7, 41.9, 1531.8, 988.3, 0.2134, -0.0871, 1.0
    )
    assert calibration_payload(calibration.stations) == expected


def test_the_int32_shim_is_the_only_quantised_path(tmp_path):
    """The shim carries a schema 2 file to firmware that still reads int32."""
    from dotbot.calibration.lighthouse2 import homography_as_bytes

    path = tmp_path / "calibration.toml"
    path.write_text(FIXTURE_TOML, encoding="utf-8")
    calibration = read_calibration_file(path)

    packed = homography_as_bytes(calibration.stations[0].matrix)
    assert len(packed) == 36
    elements = [
        int.from_bytes(packed[i : i + 4], "little", signed=True) for i in range(0, 36, 4)
    ]
    assert elements[0] == 1523400
    assert elements[8] == 1000
