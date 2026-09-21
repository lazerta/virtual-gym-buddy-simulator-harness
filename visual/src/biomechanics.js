export const BIOMECH = {
  smith_squat:{kneeFlexBottomDeg:115,trunkLeanNominalDeg:12,trunkLeanStressDeg:28,stanceHipRatio:1.15,gripShoulderRatio:1.35},
  flat_db_press:{benchDeg:0,gripShoulderRatio:1.35,elbowPlaneDeg:60,romScale:1},
  incline_db_press:{benchDeg:30,gripShoulderRatio:1.30,elbowPlaneDeg:55,romScale:1},
  barbell_bench:{benchDeg:0,gripShoulderRatio:1.40,elbowPlaneDeg:60,romScale:1},
  incline_smith_press:{benchDeg:30,gripShoulderRatio:1.35,elbowPlaneDeg:55,romScale:1},
  seated_ohp:{gripShoulderRatio:1.20,startElbowFlexDeg:95,endElbowFlexDeg:10},
  leg_press:{kneeFlexBottomDeg:100,kneeFlexTopDeg:12,railDeg:45},
  lateral_raise:{shoulderAbductionTopDeg:90,elbowFlexDeg:12},
  lat_pulldown:{gripShoulderRatio:1.20,shoulderElevationTopDeg:120,shoulderElevationBottomDeg:60},
  chest_supported_t_row:{torsoSupportDeg:35,elbowFlexBottomDeg:100,elbowPlaneDeg:35}
};

export const FORM_MODIFIERS={
  short_rom:{romScale:.60},
  knee_valgus:{valgusDeg:10},
  forward_lean:{trunkLeanDeg:28},
  elbow_flare:{extraElbowPlaneDeg:20},
  asymmetry:{sideScale:.82}
};
