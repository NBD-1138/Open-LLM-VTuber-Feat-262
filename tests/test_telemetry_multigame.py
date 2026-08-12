import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_llm_vtuber.automation.runtime_state import AssistantRuntimeController
from open_llm_vtuber.telemetry.factorio import FactorioRuntimeContext, FactorioStateAccumulator
from open_llm_vtuber.telemetry.minecraft import MinecraftRuntimeContext, MinecraftStateAccumulator
from open_llm_vtuber.telemetry.models import TelemetrySettings
from open_llm_vtuber.telemetry.msfs import MsfsRuntimeContext, MsfsStateAccumulator


def test_minecraft_accumulator_tracks_window_context_and_advancements():
    accumulator = MinecraftStateAccumulator()
    accumulator.set_runtime_context(
        MinecraftRuntimeContext(
            profile_display_name="Minecraft",
            detected_window_title="Minecraft Test World",
        )
    )

    events = accumulator.apply_log_line(
        "[12:00:00] [Render thread/INFO]: Steve has made the advancement [Stone Age]",
        TelemetrySettings(),
    )
    snapshot = accumulator.snapshot()

    assert snapshot.session["worldName"] == "Minecraft Test World"
    assert snapshot.world["lastAdvancement"] == "Stone Age"
    assert any(event.event_type == "minecraft.advancement" for event in events)
    assert any(
        highlight.label == "Last milestone" and highlight.value == "Stone Age"
        for highlight in snapshot.highlights
    )


def test_factorio_accumulator_tracks_save_and_research():
    accumulator = FactorioStateAccumulator()
    accumulator.set_runtime_context(
        FactorioRuntimeContext(
            profile_display_name="Factorio",
            detected_window_title="Factorio - Nauvis Main Bus",
        )
    )

    load_events = accumulator.apply_log_line(
        "   0.420 Info Scenario.cpp:194: Loading map C:/Saves/Nauvis Main Bus.zip (current version 2.0.0)",
        TelemetrySettings(),
    )
    research_events = accumulator.apply_log_line(
        "123.456 Info Technology.cpp:999: Research completed: Logistics 3",
        TelemetrySettings(),
    )
    snapshot = accumulator.snapshot()

    assert any(event.event_type == "factorio.save.loaded" for event in load_events)
    assert any(
        event.event_type == "factorio.research.completed"
        for event in research_events
    )
    assert snapshot.session["saveName"] == "Nauvis Main Bus"
    assert snapshot.world["lastResearch"] == "Logistics 3"
    assert any(
        highlight.label == "Last research" and highlight.value == "Logistics 3"
        for highlight in snapshot.highlights
    )


def test_msfs_accumulator_surfaces_takeoff_and_stall_warnings():
    accumulator = MsfsStateAccumulator()
    accumulator.set_runtime_context(
        MsfsRuntimeContext(
            profile_display_name="Microsoft Flight Simulator",
            detected_window_title="Microsoft Flight Simulator 2024",
        )
    )

    accumulator.apply_sample(
        {
            "title": "Cessna 172",
            "onGround": True,
            "altitudeFt": 1200.0,
            "altitudeAglFt": 0.0,
            "airspeedKts": 0.0,
            "verticalSpeedFpm": 0.0,
            "gearDown": True,
            "gearRetractable": False,
            "stallWarning": False,
            "overspeedWarning": False,
            "autopilotMaster": False,
            "nextWaypoint": "",
            "latitude": 51.0,
            "longitude": -114.0,
            "headingTrue": 0.0,
        },
        TelemetrySettings(),
    )

    events = accumulator.apply_sample(
        {
            "title": "Cessna 172",
            "onGround": False,
            "altitudeFt": 1800.0,
            "altitudeAglFt": 600.0,
            "airspeedKts": 96.0,
            "verticalSpeedFpm": 850.0,
            "gearDown": True,
            "gearRetractable": False,
            "stallWarning": True,
            "overspeedWarning": False,
            "autopilotMaster": False,
            "nextWaypoint": "CYBW",
            "latitude": 51.1,
            "longitude": -114.1,
            "headingTrue": 1.2,
        },
        TelemetrySettings(),
    )
    snapshot = accumulator.snapshot()

    event_types = {event.event_type for event in events}
    assert "msfs.takeoff" in event_types
    assert "msfs.stall.warning" in event_types
    assert snapshot.session["flightPhase"] == "departure_or_approach"
    assert any(
        highlight.label == "Airspeed" and highlight.value == "96 kts"
        for highlight in snapshot.highlights
    )


def test_runtime_prompt_prefers_generic_telemetry_highlights():
    runtime = AssistantRuntimeController()
    runtime.set_telemetry_snapshot(
        {
            "adapterId": "minecraft-log",
            "gameDisplayName": "Minecraft",
            "snapshot": {
                "stale": False,
                "confidence": 0.81,
                "sourceTimestamp": None,
                "lastUpdateTimestamp": None,
                "session": {},
                "ship": {},
                "navigation": {},
                "combat": {},
                "missions": {},
                "world": {},
                "automation": {},
                "highlights": [
                    {"label": "Session", "value": "Test World", "key": "session"},
                    {"label": "Dimension", "value": "Overworld", "key": "dimension"},
                ],
            },
        }
    )

    prompt = runtime.build_context_prompt(capabilities=[], pending_confirmations=[])
    assert "Game: Minecraft" in prompt
    assert "Session: Test World" in prompt
    assert "Dimension: Overworld" in prompt
