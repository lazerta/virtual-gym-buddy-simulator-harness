import * as THREE from "three";
import {jointFlexionDeg} from "./ik.js";
const wp=o=>o?.getWorldPosition(new THREE.Vector3())??null;
export function collectGroundTruth(model,rig,meta){
  model.updateMatrixWorld(true);const joints={};for(const[name,b]of Object.entries(rig)){if(!b)continue;const v=wp(b);joints[name]=[v.x,v.y,v.z]}
  const p=n=>wp(rig[n]);
  return {schema_version:2,timestamp_ms:performance.now(),avatar_id:meta.avatar_id,exercise_id:meta.exercise_id,form_id:meta.form_id,phase:meta.phase,issues:meta.issues,constraint_error_m:meta.constraint_error_m??0,measured:meta.measured??{},joint_angles:{left_elbow_flex_deg:jointFlexionDeg(p("leftUpperArm"),p("leftLowerArm"),p("leftHand")),right_elbow_flex_deg:jointFlexionDeg(p("rightUpperArm"),p("rightLowerArm"),p("rightHand")),left_knee_flex_deg:jointFlexionDeg(p("leftUpperLeg"),p("leftLowerLeg"),p("leftFoot")),right_knee_flex_deg:jointFlexionDeg(p("rightUpperLeg"),p("rightLowerLeg"),p("rightFoot"))},joints3d:joints};
}
