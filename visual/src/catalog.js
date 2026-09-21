import {ASSET_PATHS} from "./assets.js";

export const AVATARS={
  quaternius_human:{id:"quaternius_human",label:"CC0 Human",url:ASSET_PATHS.avatars.human,targetHeight:1.75,license:"CC0-1.0"},
  quaternius_superhero_male:{id:"quaternius_superhero_male",label:"CC0 Superhero Male",url:ASSET_PATHS.avatars.superheroMale,targetHeight:1.82,license:"CC0-1.0"}
};

export const EXERCISES={
  incline_db_press:{id:"incline_db_press",label:"Incline Dumbbell Press",equipment:"incline_bench_db"},
  flat_db_press:{id:"flat_db_press",label:"Flat Dumbbell Press",equipment:"flat_bench_db"},
  barbell_bench:{id:"barbell_bench",label:"Barbell Bench Press",equipment:"flat_bench_barbell"},
  incline_smith_press:{id:"incline_smith_press",label:"Incline Smith Press",equipment:"incline_smith"},
  seated_ohp:{id:"seated_ohp",label:"Seated Overhead Press",equipment:"ohp_machine"},
  smith_squat:{id:"smith_squat",label:"Smith Squat",equipment:"smith"},
  leg_press:{id:"leg_press",label:"Leg Press",equipment:"leg_press"},
  lateral_raise:{id:"lateral_raise",label:"Lateral Raise",equipment:"dumbbells"},
  lat_pulldown:{id:"lat_pulldown",label:"Lat Pulldown",equipment:"lat_pulldown"},
  chest_supported_t_row:{id:"chest_supported_t_row",label:"Chest-Supported T Row",equipment:"t_row"}
};

export const FORMS={
  correct:{id:"correct",label:"Correct"},
  knee_valgus:{id:"knee_valgus",label:"Knee valgus"},
  short_rom:{id:"short_rom",label:"Short ROM"},
  forward_lean:{id:"forward_lean",label:"Forward lean"},
  elbow_flare:{id:"elbow_flare",label:"Elbow flare"},
  asymmetry:{id:"asymmetry",label:"Left/right asymmetry"}
};
