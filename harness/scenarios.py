from __future__ import annotations
from dataclasses import replace
import numpy as np
from .models import Scenario

FAMILIES = [
 "clean","setup_motion","wrong_view","too_close","too_far","camera_low","camera_high","camera_bump",
 "bystander_cross","bystander_bg","spotter","foreground_occlusion","subject_swap","lookalike_competitor",
 "mirror_known","mirror_ambiguous","motion_blur","low_light","tracking_gap","single_issue",
 "repeated_issue","short_rom","asymmetry","target_exit_reenter"
]

def deterministic_scenario(family: str, seed: int) -> Scenario:
    s=Scenario(id=f"{family}-{seed}", family=family, seed=seed)
    if family=="setup_motion": return replace(s, setup_motion=True, rep_count_gt=0)
    if family=="wrong_view": return replace(s, camera_yaw_deg=120)
    if family=="too_close": return replace(s, camera_distance_m=1.7)
    if family=="too_far": return replace(s, camera_distance_m=10.0)
    if family=="camera_low": return replace(s, camera_height_m=.45)
    if family=="camera_high": return replace(s, camera_height_m=2.7)
    if family=="camera_bump": return replace(s, camera_bump_deg=12)
    if family=="bystander_cross": return replace(s, bystander_count=1, competitor_similarity=.22)
    if family=="bystander_bg": return replace(s, bystander_count=2, competitor_similarity=.18)
    if family=="spotter": return replace(s, bystander_count=1, competitor_similarity=.30, target_occlusion_fraction=.06)
    if family=="foreground_occlusion": return replace(s, bystander_count=1, competitor_similarity=.20, target_occlusion_fraction=.48, occlusion_fraction=.48, rep_count_gt=9)
    if family=="subject_swap": return replace(s, identity_jump=True, bystander_count=1, competitor_similarity=.88, rep_count_gt=0)
    if family=="lookalike_competitor": return replace(s, bystander_count=1, competitor_similarity=.97, rep_count_gt=0)
    if family=="mirror_known": return replace(s, bystander_count=1, mirror_known_region=True, competitor_similarity=.82)
    if family=="mirror_ambiguous": return replace(s, bystander_count=1, mirror_ambiguity=True, competitor_similarity=.88, rep_count_gt=0)
    if family=="motion_blur": return replace(s, tracking_quality=.48, rep_count_gt=0)
    if family=="low_light": return replace(s, tracking_quality=.48, rep_count_gt=0)
    if family=="tracking_gap": return replace(s, tracking_gap_ms=420, rep_count_gt=9)
    if family=="target_exit_reenter": return replace(s, tracking_gap_ms=900, rep_count_gt=9)
    if family=="single_issue": return replace(s, issue_strength=.75, repeated_issue_reps=(6,))
    if family=="repeated_issue": return replace(s, issue_strength=.75, repeated_issue_reps=(7,8))
    if family=="short_rom": return replace(s, issue_strength=.68, repeated_issue_reps=(8,9,10))
    if family=="asymmetry": return replace(s, issue_strength=.72, repeated_issue_reps=(7,9))
    return s

def fuzz_scenario(rng: np.random.Generator, seed: int) -> Scenario:
    family=str(rng.choice(FAMILIES))
    s=deterministic_scenario(family, seed)
    dist=max(.8, s.camera_distance_m + rng.normal(0,.23))
    yaw=s.camera_yaw_deg + rng.normal(0,4.0)
    height=max(.25, s.camera_height_m + rng.normal(0,.12))
    tq=float(np.clip(s.tracking_quality + rng.normal(0,.075),0,1))
    occ=float(np.clip(s.occlusion_fraction + rng.normal(0,.05),0,1))
    target_occ=float(np.clip(s.target_occlusion_fraction + rng.normal(0,.04),0,1))
    competitor=float(np.clip(s.competitor_similarity + rng.normal(0,.055),0,1))
    bump=max(0., s.camera_bump_deg + rng.normal(0,1.8)) if family=="camera_bump" else 0.
    issue=float(np.clip(s.issue_strength + rng.normal(0,.09),0,1))
    return replace(s,camera_distance_m=dist,camera_yaw_deg=yaw,camera_height_m=height,
        tracking_quality=tq,occlusion_fraction=occ,target_occlusion_fraction=target_occ,
        competitor_similarity=competitor,camera_bump_deg=bump,issue_strength=issue)

COMPOSITE_GROUPS = {
    "camera_geometry": ["wrong_view","too_close","too_far","camera_low","camera_high"],
    "camera_runtime": ["camera_bump"],
    "people": ["bystander_cross","bystander_bg","spotter","foreground_occlusion","lookalike_competitor","subject_swap","mirror_known","mirror_ambiguous"],
    "observation": ["motion_blur","low_light","tracking_gap","target_exit_reenter"],
    "movement": ["single_issue","repeated_issue","short_rom","asymmetry"],
}

def _with_components(s: Scenario, comps: tuple[str,...]) -> Scenario:
    return replace(s, family="+".join(comps), id=f"{'+'.join(comps)}-{s.seed}", components=comps)

def compose_scenario(families: list[str] | tuple[str,...], seed: int) -> Scenario:
    comps=tuple(dict.fromkeys(families))
    if not comps:
        return replace(deterministic_scenario("clean",seed),components=("clean",))
    cam=[x for x in comps if x in COMPOSITE_GROUPS["camera_geometry"]]
    if len(cam)>1:
        raise ValueError(f"contradictory camera geometry components: {cam}")

    base=Scenario(id=f"composite-{seed}",family="composite",seed=seed,components=comps)
    repeated=set(); issue_strength=0.0
    vals={
        "camera_distance_m":base.camera_distance_m,"camera_yaw_deg":base.camera_yaw_deg,
        "camera_height_m":base.camera_height_m,"bystander_count":base.bystander_count,
        "identity_jump":False,"mirror_ambiguity":False,"mirror_known_region":False,
        "competitor_similarity":base.competitor_similarity,"target_occlusion_fraction":0.0,
        "occlusion_fraction":0.0,"tracking_quality":base.tracking_quality,
        "tracking_gap_ms":0,"camera_bump_deg":0.0,"setup_motion":False,
        "rep_count_gt":10,
    }
    for fam in comps:
        x=deterministic_scenario(fam,seed)
        if fam in {"too_close","too_far"}: vals["camera_distance_m"]=x.camera_distance_m
        if fam=="wrong_view": vals["camera_yaw_deg"]=x.camera_yaw_deg
        if fam in {"camera_low","camera_high"}: vals["camera_height_m"]=x.camera_height_m
        vals["bystander_count"]+=x.bystander_count
        vals["identity_jump"] = vals["identity_jump"] or x.identity_jump
        vals["mirror_ambiguity"] = vals["mirror_ambiguity"] or x.mirror_ambiguity
        vals["mirror_known_region"] = vals["mirror_known_region"] or x.mirror_known_region
        vals["competitor_similarity"] = max(vals["competitor_similarity"],x.competitor_similarity)
        vals["target_occlusion_fraction"] = max(vals["target_occlusion_fraction"],x.target_occlusion_fraction)
        vals["occlusion_fraction"] = max(vals["occlusion_fraction"],x.occlusion_fraction)
        vals["tracking_quality"] = min(vals["tracking_quality"],x.tracking_quality)
        vals["tracking_gap_ms"] = max(vals["tracking_gap_ms"],x.tracking_gap_ms)
        vals["camera_bump_deg"] = max(vals["camera_bump_deg"],x.camera_bump_deg)
        vals["setup_motion"] = vals["setup_motion"] or x.setup_motion
        vals["rep_count_gt"] = min(vals["rep_count_gt"], x.rep_count_gt)
        issue_strength=max(issue_strength,x.issue_strength)
        repeated.update(x.repeated_issue_reps)
    s=replace(base,**vals,issue_strength=issue_strength,repeated_issue_reps=tuple(sorted(repeated)))
    return _with_components(s,comps)

def fuzz_composite_scenario(rng: np.random.Generator, seed: int, max_components: int = 3) -> Scenario:
    groups=list(COMPOSITE_GROUPS)
    k=int(rng.integers(2,max_components+1))
    chosen_groups=list(rng.choice(groups,size=k,replace=False))
    comps=[str(rng.choice(COMPOSITE_GROUPS[g])) for g in chosen_groups]
    s=compose_scenario(comps,seed)
    dist=max(.8,s.camera_distance_m+rng.normal(0,.18))
    yaw=s.camera_yaw_deg+rng.normal(0,3.0)
    height=max(.25,s.camera_height_m+rng.normal(0,.10))
    tq=float(np.clip(s.tracking_quality+rng.normal(0,.055),0,1))
    occ=float(np.clip(s.occlusion_fraction+rng.normal(0,.035),0,1))
    tocc=float(np.clip(s.target_occlusion_fraction+rng.normal(0,.035),0,1))
    comp=float(np.clip(s.competitor_similarity+rng.normal(0,.04),0,1))
    issue=float(np.clip(s.issue_strength+rng.normal(0,.07),0,1))
    bump=max(0.,s.camera_bump_deg+rng.normal(0,1.2)) if s.camera_bump_deg else 0.0
    return replace(s,camera_distance_m=dist,camera_yaw_deg=yaw,camera_height_m=height,
                   tracking_quality=tq,occlusion_fraction=occ,target_occlusion_fraction=tocc,
                   competitor_similarity=comp,issue_strength=issue,camera_bump_deg=bump)
