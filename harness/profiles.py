from __future__ import annotations
from .models import SubjectProfile, ExerciseProfile

EXERCISES = [
    ExerciseProfile("incline_db_press", "press_db", 32, 20, 0.42, 0.86, True),
    ExerciseProfile("flat_db_press", "press_db", 35, 22, 0.42, 0.86, True),
    ExerciseProfile("barbell_bench", "press_bar", 35, 20, 0.43, 0.88, False),
    ExerciseProfile("incline_smith_press", "press_smith", 30, 18, 0.42, 0.86, False),
    ExerciseProfile("seated_ohp", "press_machine", 25, 20, 0.44, 0.88, True),
    ExerciseProfile("smith_squat", "squat", 78, 16, 0.55, 0.94, False),
    ExerciseProfile("leg_press", "leg_press", 55, 25, 0.48, 0.92, False),
    ExerciseProfile("lateral_raise", "raise", 5, 18, 0.55, 0.93, True),
    ExerciseProfile("lat_pulldown", "vertical_pull", 8, 22, 0.52, 0.92, True),
    ExerciseProfile("chest_supported_t_row", "row", 45, 25, 0.48, 0.90, True),
]

# Anonymous stress fixtures derived around published anthropometric ranges.
# They are not literal individual ANSUR records.
def external_subjects() -> list[SubjectProfile]:
    anchors = [
        ("small", .92, .305, .180, .142, .238, .242, .238, .181),
        ("lower_mid", .96, .308, .183, .144, .241, .244, .241, .183),
        ("median", 1.00, .310, .186, .146, .245, .246, .245, .185),
        ("upper_mid", 1.04, .312, .189, .148, .248, .248, .249, .188),
        ("large", 1.08, .314, .192, .150, .252, .250, .253, .191),
        ("very_large", 1.12, .316, .195, .152, .255, .252, .257, .194),
    ]
    out=[]
    for name,scale,torso,ua,fa,th,sh,sw,hw in anchors:
        out.append(SubjectProfile(f"ansur_{name}_central","ANSUR-II-derived-stress",scale,torso,ua,fa,th,sh,sw,hw,scale))
        out.append(SubjectProfile(f"ansur_{name}_long_limb","ANSUR-II-derived-stress",scale,torso*.94,ua*1.045,fa*1.045,th*1.035,sh*1.035,sw,hw,scale))
        out.append(SubjectProfile(f"ansur_{name}_long_torso","ANSUR-II-derived-stress",scale,torso*1.06,ua*.965,fa*.965,th*.975,sh*.975,sw,hw,scale))
    return out

SHAWN_LIKE = SubjectProfile(
    id="shawn_like_v0_1", source="user-reference-images-bootstrap",
    stature_scale=1.0, torso_ratio=.312, upper_arm_ratio=.186, forearm_ratio=.145,
    thigh_ratio=.244, shin_ratio=.246, shoulder_width_ratio=.249,
    hip_width_ratio=.184, silhouette_scale=1.0, visual_variant="current_like"
)

SHAWN_VARIANTS = [
    SHAWN_LIKE,
    SubjectProfile(**{**SHAWN_LIKE.__dict__, "id":"shawn_like_loose_clothes", "silhouette_scale":1.06, "visual_variant":"loose_clothes"}),
    SubjectProfile(**{**SHAWN_LIKE.__dict__, "id":"shawn_like_fitted", "silhouette_scale":.98, "visual_variant":"fitted"}),
    SubjectProfile(**{**SHAWN_LIKE.__dict__, "id":"shawn_like_leaner", "silhouette_scale":.94, "visual_variant":"leaner"}),
    SubjectProfile(**{**SHAWN_LIKE.__dict__, "id":"shawn_like_heavier", "silhouette_scale":1.08, "visual_variant":"heavier"}),
    SubjectProfile(**{**SHAWN_LIKE.__dict__, "id":"shawn_like_glasses_hair", "visual_variant":"glasses_high_hair"}),
]
