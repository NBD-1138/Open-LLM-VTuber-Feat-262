import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.automation.runtime_state import AssistantRuntimeController
from open_llm_vtuber.telemetry import TelemetryAdapterSnapshot, TelemetrySettings
from open_llm_vtuber.telemetry import elite_dangerous as elite_module
from open_llm_vtuber.telemetry.elite_dangerous import (
    STALE_AFTER_SECONDS,
    EliteDangerousReplayHarness,
    EliteDangerousStateAccumulator,
    EliteDangerousTelemetryAdapter,
    JournalTailer,
    StatusReader,
    _utcnow,
    discover_elite_journal_directory,
)
from open_llm_vtuber.telemetry.manager import ClientTelemetryRuntime


def _run(coro):
    return asyncio.run(coro)


def _flags(*bits: int) -> int:
    value = 0
    for bit in bits:
        value |= 1 << bit
    return value


def test_discover_elite_journal_directory_supports_auto_and_manual_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    automatic_dir = (
        tmp_path / "Saved Games" / "Frontier Developments" / "Elite Dangerous"
    )
    automatic_dir.mkdir(parents=True)
    manual_dir = tmp_path / "Manual Elite"
    manual_dir.mkdir()

    monkeypatch.setattr(elite_module, "DEFAULT_JOURNAL_DIR", automatic_dir)

    automatic_path, automatic_status, automatic_label = (
        discover_elite_journal_directory(TelemetrySettings())
    )
    assert automatic_path == automatic_dir
    assert automatic_status == "automatic"
    assert automatic_label

    manual_path, manual_status, manual_label = discover_elite_journal_directory(
        TelemetrySettings(
            automatic_journal_discovery=False,
            manual_journal_directory=str(manual_dir),
        )
    )
    assert manual_path == manual_dir
    assert manual_status == "manual"
    assert manual_label

    missing_path, missing_status, _missing_label = discover_elite_journal_directory(
        TelemetrySettings(
            automatic_journal_discovery=False,
            manual_journal_directory=str(tmp_path / "missing"),
        )
    )
    assert missing_path == tmp_path / "missing"
    assert missing_status == "missing"


def test_journal_tailer_handles_partial_lines_and_malformed_entries(tmp_path: Path):
    journal_dir = tmp_path / "Elite Dangerous"
    journal_dir.mkdir()
    journal_file = journal_dir / "Journal.2026-07-20T120000.01.log"

    first_entry = {"timestamp": "2026-07-20T12:00:00Z", "event": "LoadGame"}
    second_entry = {"timestamp": "2026-07-20T12:00:01Z", "event": "Location"}
    third_entry = {"timestamp": "2026-07-20T12:00:02Z", "event": "DockingGranted"}

    journal_file.write_text(
        json.dumps(first_entry) + "\n" + json.dumps(second_entry)[:24],
        encoding="utf-8",
    )

    tailer = JournalTailer()
    first_poll = _run(tailer.poll(journal_dir))
    assert first_poll.current_file == journal_file.name
    assert first_poll.parse_errors == 0
    assert [event["event"] for event in first_poll.events] == ["LoadGame"]

    with journal_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(second_entry)[24:] + "\n")
        handle.write("{bad json}\n")
        handle.write(json.dumps(third_entry) + "\n")

    second_poll = _run(tailer.poll(journal_dir))
    assert second_poll.parse_errors == 1
    assert [event["event"] for event in second_poll.events] == [
        "Location",
        "DockingGranted",
    ]

    third_poll = _run(tailer.poll(journal_dir))
    assert third_poll.events == []


def test_journal_tailer_switches_to_newer_journal_and_reads_only_appended_lines(
    tmp_path: Path,
):
    journal_dir = tmp_path / "Elite Dangerous"
    journal_dir.mkdir()
    journal_one = journal_dir / "Journal.2026-07-20T120000.01.log"
    journal_two = journal_dir / "Journal.2026-07-20T130000.01.log"

    load_game = {"timestamp": "2026-07-20T12:00:00Z", "event": "LoadGame"}
    location = {"timestamp": "2026-07-20T12:00:01Z", "event": "Location"}
    jump = {"timestamp": "2026-07-20T13:00:00Z", "event": "FSDJump"}

    journal_one.write_text(json.dumps(load_game) + "\n", encoding="utf-8")
    tailer = JournalTailer()

    first_poll = _run(tailer.poll(journal_dir))
    assert [event["event"] for event in first_poll.events] == ["LoadGame"]

    second_poll = _run(tailer.poll(journal_dir))
    assert second_poll.events == []

    with journal_one.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(location) + "\n")

    third_poll = _run(tailer.poll(journal_dir))
    assert [event["event"] for event in third_poll.events] == ["Location"]

    journal_two.write_text(json.dumps(jump) + "\n", encoding="utf-8")
    newer_mtime = journal_one.stat().st_mtime + 10
    os.utime(journal_two, (newer_mtime, newer_mtime))

    rotated_poll = _run(tailer.poll(journal_dir))
    assert rotated_poll.current_file == journal_two.name
    assert [event["event"] for event in rotated_poll.events] == ["FSDJump"]


def test_status_reader_debounces_unchanged_status_file(tmp_path: Path):
    journal_dir = tmp_path / "Elite Dangerous"
    journal_dir.mkdir()
    status_file = journal_dir / "Status.json"
    status_file.write_text(
        json.dumps({"timestamp": "2026-07-20T12:00:00Z", "Flags": _flags(3)}),
        encoding="utf-8",
    )

    reader = StatusReader()
    first_poll = _run(reader.poll(journal_dir))
    assert first_poll["Flags"] == _flags(3)

    second_poll = _run(reader.poll(journal_dir))
    assert second_poll is None

    status_file.write_text(
        json.dumps({"timestamp": "2026-07-20T12:00:02Z", "Flags": _flags(22)}),
        encoding="utf-8",
    )
    newer_mtime = status_file.stat().st_mtime + 10
    os.utime(status_file, (newer_mtime, newer_mtime))

    third_poll = _run(reader.poll(journal_dir))
    assert third_poll["Flags"] == _flags(22)


def test_status_flags_update_structured_state_and_surface_emergency_events():
    accumulator = EliteDangerousStateAccumulator()
    settings = TelemetrySettings()

    accumulator.apply_status(
        {"timestamp": "2026-07-20T12:00:00Z", "Flags": _flags(3)},
        settings,
    )
    events = accumulator.apply_status(
        {"timestamp": "2026-07-20T12:00:02Z", "Flags": _flags(6, 20, 22)},
        settings,
    )
    event_types = {event.event_type for event in events}

    assert "elite.shields.failed" in event_types
    assert "elite.danger.started" in event_types
    assert "elite.combat.entered" in event_types
    assert "elite.heat.warning" in event_types
    assert all(event.importance >= 8 for event in events if event.high_priority)

    snapshot = accumulator.snapshot()
    assert snapshot.ship["hardpointsDeployed"] is True
    assert snapshot.ship["shieldsUp"] is False
    assert snapshot.combat["inDanger"] is True

    accumulator._last_update_at = _utcnow() - timedelta(seconds=STALE_AFTER_SECONDS + 2)
    stale_snapshot = accumulator.snapshot()
    assert stale_snapshot.stale is True
    assert stale_snapshot.confidence < snapshot.confidence


def test_hull_threshold_events_only_fire_on_downward_crossings_and_reset_after_respawn():
    accumulator = EliteDangerousStateAccumulator()
    settings = TelemetrySettings(hull_warning_thresholds=[75, 50, 25])

    accumulator.apply_journal_entry(
        {
            "timestamp": "2026-07-20T12:00:00Z",
            "event": "LoadGame",
            "GameMode": "Solo",
            "Ship": "AspExplorer",
        },
        settings,
    )

    assert (
        accumulator.apply_journal_entry(
            {
                "timestamp": "2026-07-20T12:01:00Z",
                "event": "HullDamage",
                "Health": 0.80,
            },
            settings,
        )
        == []
    )

    first_warning = accumulator.apply_journal_entry(
        {"timestamp": "2026-07-20T12:01:05Z", "event": "HullDamage", "Health": 0.74},
        settings,
    )
    assert [event.novelty_key for event in first_warning] == ["elite.hull.threshold:75"]

    assert (
        accumulator.apply_journal_entry(
            {
                "timestamp": "2026-07-20T12:01:10Z",
                "event": "HullDamage",
                "Health": 0.70,
            },
            settings,
        )
        == []
    )

    second_warning = accumulator.apply_journal_entry(
        {"timestamp": "2026-07-20T12:01:20Z", "event": "HullDamage", "Health": 0.49},
        settings,
    )
    assert [event.novelty_key for event in second_warning] == [
        "elite.hull.threshold:50"
    ]

    accumulator.apply_journal_entry(
        {"timestamp": "2026-07-20T12:02:00Z", "event": "Resurrect"},
        settings,
    )
    reset_warning = accumulator.apply_journal_entry(
        {"timestamp": "2026-07-20T12:02:05Z", "event": "HullDamage", "Health": 0.74},
        settings,
    )
    assert [event.novelty_key for event in reset_warning] == ["elite.hull.threshold:75"]


def test_fuel_warning_rearms_only_after_recovery():
    accumulator = EliteDangerousStateAccumulator()
    settings = TelemetrySettings(low_fuel_threshold=0.25, critical_fuel_threshold=0.10)

    accumulator.apply_journal_entry(
        {
            "timestamp": "2026-07-20T12:00:00Z",
            "event": "LoadGame",
            "GameMode": "Solo",
            "FuelLevel": 16.0,
            "FuelCapacity": {"Main": 32.0},
        },
        settings,
    )

    first_warning = accumulator.apply_status(
        {
            "timestamp": "2026-07-20T12:01:00Z",
            "Flags": _flags(),
            "Fuel": {"FuelMain": 7.0},
        },
        settings,
    )
    assert [event.summary for event in first_warning] == ["Fuel is running low."]

    assert (
        accumulator.apply_status(
            {
                "timestamp": "2026-07-20T12:01:05Z",
                "Flags": _flags(),
                "Fuel": {"FuelMain": 6.0},
            },
            settings,
        )
        == []
    )

    accumulator.apply_status(
        {
            "timestamp": "2026-07-20T12:01:10Z",
            "Flags": _flags(),
            "Fuel": {"FuelMain": 12.0},
        },
        settings,
    )

    critical_warning = accumulator.apply_status(
        {
            "timestamp": "2026-07-20T12:01:15Z",
            "Flags": _flags(),
            "Fuel": {"FuelMain": 3.0},
        },
        settings,
    )
    assert [event.summary for event in critical_warning] == ["Fuel is critically low."]


def test_capability_advice_uses_telemetry_state():
    adapter = EliteDangerousTelemetryAdapter(
        on_event=lambda _event: asyncio.sleep(0),
        on_state_change=lambda: asyncio.sleep(0),
    )

    _run(
        adapter.process_journal_entry(
            {
                "timestamp": "2026-07-20T12:00:00Z",
                "event": "Location",
                "StarSystem": "Shinrarta Dezhra",
                "StationName": "Jameson Memorial",
                "Docked": False,
            }
        )
    )
    _run(
        adapter.process_status_payload(
            {"timestamp": "2026-07-20T12:00:02Z", "Flags": _flags(6, 22)}
        )
    )
    _run(
        adapter.process_journal_entry(
            {"timestamp": "2026-07-20T12:00:05Z", "event": "DockingGranted"}
        )
    )

    advice = adapter.build_capability_advice(
        [
            {"command_id": "deploy_hardpoints", "label": "Deploy Hardpoints"},
            {"command_id": "retract_hardpoints", "label": "Retract Hardpoints"},
            {"command_id": "landing_gear", "label": "Landing Gear"},
            {"command_id": "request_docking", "label": "Request Docking"},
        ]
    )

    assert advice["deploy_hardpoints"].available_override is False
    assert (
        advice["deploy_hardpoints"].blocked_reason == "Hardpoints are already deployed."
    )
    assert (
        advice["retract_hardpoints"].avoid_reason
        == "Retracting hardpoints during danger is discouraged."
    )
    assert advice["landing_gear"].recommended is True
    assert advice["request_docking"].blocked_reason == "Docking is already granted."


def test_client_runtime_starts_and_stops_matching_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeAdapter:
        supported_process_names = ("elitedangerous64.exe",)
        instances = []

        def __init__(self, _on_event, _on_state_change):
            self.start_calls = []
            self.stop_calls = 0
            FakeAdapter.instances.append(self)

        async def start(self, settings):
            self.start_calls.append(settings)

        async def stop(self):
            self.stop_calls += 1

        def current_snapshot(self):
            return TelemetryAdapterSnapshot(
                adapter_id="elite-dangerous-journal",
                game_id="elite-dangerous",
                status="running",
            )

        def build_capability_advice(self, _capabilities):
            return {}

    monkeypatch.setattr(elite_module, "EliteDangerousTelemetryAdapter", FakeAdapter)
    monkeypatch.setattr(
        "open_llm_vtuber.telemetry.manager.EliteDangerousTelemetryAdapter", FakeAdapter
    )

    runtime = ClientTelemetryRuntime(
        on_event=lambda _event: asyncio.sleep(0),
        on_state_change=lambda: asyncio.sleep(0),
    )

    _run(runtime.update_settings(TelemetrySettings().to_payload()))
    _run(
        runtime.update_context(
            profile_id="elite-dangerous",
            profile_process_names=["EliteDangerous64.exe"],
            detected_process_name="EliteDangerous64.exe",
        )
    )
    _run(
        runtime.update_context(
            profile_id="elite-dangerous",
            profile_process_names=["EliteDangerous64.exe"],
            detected_process_name="EliteDangerous64.exe",
        )
    )
    assert len(FakeAdapter.instances) == 1
    assert len(FakeAdapter.instances[0].start_calls) >= 1

    _run(
        runtime.update_context(
            profile_id="other-game",
            profile_process_names=["other.exe"],
            detected_process_name="other.exe",
        )
    )
    assert FakeAdapter.instances[0].stop_calls == 1
    assert runtime.snapshot_payload()["status"] == "stopped"


def test_replay_harness_feeds_the_same_parser_pipeline_and_restores_replay_mode():
    events = []
    state_changes = []

    async def on_event(event):
        events.append(event.event_type)

    async def on_state_change():
        state_changes.append("changed")

    adapter = EliteDangerousTelemetryAdapter(
        on_event=on_event, on_state_change=on_state_change
    )
    harness = EliteDangerousReplayHarness(adapter)

    journal_lines = [
        json.dumps(
            {
                "timestamp": "2026-07-20T12:00:00Z",
                "event": "LoadGame",
                "GameMode": "Open",
                "Commander": "Test Cmdr",
                "Ship": "CobraMkIII",
            }
        ),
        "{bad json}",
        json.dumps(
            {
                "timestamp": "2026-07-20T12:00:01Z",
                "event": "Location",
                "StarSystem": "Sol",
            }
        ),
    ]
    status_snapshots = [
        {"timestamp": "2026-07-20T12:00:00Z", "Flags": _flags(3)},
        {"timestamp": "2026-07-20T12:00:01Z", "Flags": _flags()},
    ]

    _run(
        harness.replay(
            journal_lines=journal_lines,
            status_snapshots=status_snapshots,
            speed=0,
        )
    )

    assert "elite.session.started" in events
    assert "elite.location.changed" in events
    assert "elite.shields.failed" in events
    assert state_changes
    assert adapter.current_snapshot().replay_mode is False
    assert adapter.current_snapshot().status == "stopped"


def test_runtime_prompt_includes_compact_telemetry_without_raw_paths():
    runtime = AssistantRuntimeController()
    runtime.set_telemetry_snapshot(
        {
            "adapterId": "elite-dangerous-journal",
            "journalDirectoryLabel": "Elite Dangerous",
            "currentJournalFile": "Journal.2026-07-20T120000.01.log",
            "snapshot": {
                "stale": False,
                "session": {
                    "starSystem": "Sol",
                    "shipName": "Asp Explorer",
                    "docked": False,
                },
                "ship": {
                    "hullPercent": 87.5,
                    "fuelLevel": 14.2,
                },
                "combat": {"inDanger": True},
            },
        }
    )

    prompt = runtime.build_context_prompt(capabilities=[], pending_confirmations=[])
    assert "GAME TELEMETRY" in prompt
    assert "Sol" in prompt
    assert "Asp Explorer" in prompt
    assert "C:\\Users\\morga\\Saved Games" not in prompt
    assert "Frontier Developments" not in prompt
