from __future__ import annotations

from dataclasses import dataclass, replace
from .models import Scenario
from .scenarios import compose_scenario


@dataclass(frozen=True)
class CommercialGymScene:
    """A sensor-level commercial-gym environment manifest.

    The current P0 harness does not try to render photorealistic RGB. Instead it
    models the environmental factors that matter to subject association and movement
    observability.  The same manifest can later drive Blender/RGB rendering.

    ambient_noise is metadata only because the current Gym Buddy design does not use
    microphone input for exercise control. Acoustic noise should therefore not change
    biomechanics decisions.
    """

    id: str
    components: tuple[str, ...]
    crowd_level: str
    mirror_level: str
    lighting: str
    equipment_clutter: str
    ambient_noise: str
    moving_people: int = 0
    stationary_people: int = 0
    visual_clutter: float = 0.0
    background_motion: float = 0.0
    equipment_occlusion_fraction: float = 0.0
    camera_micro_jitter: float = 0.0
    lighting_instability: float = 0.0


COMMERCIAL_GYM_SCENES: tuple[CommercialGymScene, ...] = (
    CommercialGymScene("off_peak_clean", (), "low", "none", "normal", "low", "moderate", 0, 1, .05, .04, .01, .01, .01),
    CommercialGymScene("off_peak_background", ("bystander_bg",), "low", "none", "normal", "low", "moderate", 1, 1, .08, .08, .01, .01, .01),
    CommercialGymScene("moderate_background", ("bystander_bg",), "medium", "none", "normal", "medium", "loud", 2, 2, .16, .18, .03, .015, .02),
    CommercialGymScene("busy_background", ("bystander_cross",), "high", "none", "normal", "high", "loud", 4, 3, .28, .32, .05, .02, .02),
    CommercialGymScene("very_busy_no_occlusion", ("bystander_bg",), "high", "none", "normal", "high", "very_loud", 5, 5, .34, .40, .06, .025, .02),
    CommercialGymScene("bench_spotter", ("spotter",), "medium", "none", "normal", "medium", "loud", 1, 2, .14, .12, .03, .01, .01),
    CommercialGymScene("close_spotter_busy", ("spotter","bystander_bg"), "high", "none", "normal", "high", "very_loud", 4, 3, .28, .30, .06, .02, .02),
    CommercialGymScene("spotter_with_real_issue", ("spotter","repeated_issue"), "medium", "none", "normal", "medium", "loud", 1, 2, .14, .12, .03, .01, .01),
    CommercialGymScene("mirror_wall", ("mirror_known",), "medium", "high", "normal", "medium", "loud", 2, 2, .18, .18, .03, .015, .02),
    CommercialGymScene("mirror_wall_busy", ("mirror_known","bystander_bg"), "high", "high", "normal", "high", "very_loud", 4, 4, .30, .34, .05, .02, .02),
    CommercialGymScene("ambiguous_reflection", ("mirror_ambiguous",), "medium", "high", "normal", "medium", "loud", 2, 2, .18, .18, .03, .015, .02),
    CommercialGymScene("lookalike_competitor", ("lookalike_competitor",), "high", "none", "normal", "high", "loud", 3, 3, .28, .30, .05, .02, .02),
    CommercialGymScene("foreground_occlusion", ("foreground_occlusion",), "high", "none", "normal", "high", "loud", 4, 3, .28, .30, .04, .02, .02),
    CommercialGymScene("machine_frame_partial_occlusion", (), "medium", "none", "normal", "high", "loud", 2, 2, .22, .20, .10, .015, .02),
    CommercialGymScene("rack_and_plate_clutter", (), "high", "none", "normal", "high", "loud", 3, 4, .30, .28, .08, .015, .02),
    CommercialGymScene("tripod_floor_vibration", (), "medium", "none", "normal", "medium", "loud", 2, 2, .18, .18, .03, .12, .02),
    CommercialGymScene("tripod_bump_in_crowd", ("camera_bump","bystander_bg"), "high", "none", "normal", "high", "very_loud", 4, 4, .30, .34, .05, .04, .02),
    CommercialGymScene("motion_blur_busy", ("motion_blur","bystander_bg"), "high", "none", "normal", "high", "loud", 4, 3, .30, .38, .05, .02, .02),
    CommercialGymScene("low_light_busy", ("low_light","bystander_bg"), "high", "none", "low", "high", "loud", 4, 3, .30, .34, .05, .02, .16),
    CommercialGymScene("variable_lighting_busy", ("bystander_bg",), "high", "none", "variable", "high", "loud", 4, 3, .30, .34, .05, .02, .10),
    CommercialGymScene("tracking_gap_spotter", ("tracking_gap","spotter"), "medium", "none", "normal", "medium", "loud", 1, 2, .16, .16, .03, .01, .02),
    CommercialGymScene("leave_reenter_busy", ("target_exit_reenter","bystander_bg"), "high", "none", "normal", "high", "very_loud", 5, 4, .32, .38, .05, .02, .02),
    CommercialGymScene("mirror_real_issue", ("mirror_known","repeated_issue"), "medium", "high", "normal", "medium", "loud", 2, 2, .18, .18, .03, .015, .02),
    CommercialGymScene("busy_real_issue", ("bystander_bg","repeated_issue"), "high", "none", "normal", "high", "very_loud", 4, 4, .30, .36, .05, .02, .02),
)


def scenario_from_gym_scene(scene: CommercialGymScene, seed: int) -> Scenario:
    s=compose_scenario(scene.components, seed)
    people=max(s.bystander_count, scene.moving_people + scene.stationary_people)
    return replace(
        s,
        bystander_count=people,
        visual_clutter=scene.visual_clutter,
        background_motion=scene.background_motion,
        equipment_occlusion_fraction=scene.equipment_occlusion_fraction,
        camera_micro_jitter=scene.camera_micro_jitter,
        lighting_instability=scene.lighting_instability,
    )
