import * as THREE from "three";
import {jointFlexionDeg} from "./ik.js";

const wp=o=>o?.getWorldPosition(new THREE.Vector3())??null;
const finite=v=>Number.isFinite(v);

function projectJoint(world,camera,width,height){
  const ndc=world.clone().project(camera);
  return {
    x_px:(ndc.x*.5+.5)*width,
    y_px:(-.5*ndc.y+.5)*height,
    ndc:[ndc.x,ndc.y,ndc.z],
    in_frame:ndc.x>=-1&&ndc.x<=1&&ndc.y>=-1&&ndc.y<=1&&ndc.z>=-1&&ndc.z<=1
  };
}

export function collectGroundTruth(model,rig,meta,camera,renderer){
  model.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  const size=new THREE.Vector2();
  renderer.getSize(size);
  const joints3d={},joints2d={};
  for(const[name,b]of Object.entries(rig)){
    if(!b)continue;
    const v=wp(b);
    if(![v.x,v.y,v.z].every(finite))continue;
    joints3d[name]=[v.x,v.y,v.z];
    joints2d[name]=projectJoint(v,camera,size.x,size.y);
  }
  const p=n=>wp(rig[n]);
  const cameraWorld=camera.matrixWorld.clone();
  const worldToCamera=camera.matrixWorldInverse.clone();
  return{
    schema_version:3,
    timestamp_ms:performance.now(),
    avatar_id:meta.avatar_id,
    exercise_id:meta.exercise_id,
    form_id:meta.form_id,
    phase:meta.phase,
    issues:meta.issues,
    constraint_error_m:meta.constraint_error_m??0,
    measured:meta.measured??{},
    joint_angles:{
      left_elbow_flex_deg:jointFlexionDeg(p("leftUpperArm"),p("leftLowerArm"),p("leftHand")),
      right_elbow_flex_deg:jointFlexionDeg(p("rightUpperArm"),p("rightLowerArm"),p("rightHand")),
      left_knee_flex_deg:jointFlexionDeg(p("leftUpperLeg"),p("leftLowerLeg"),p("leftFoot")),
      right_knee_flex_deg:jointFlexionDeg(p("rightUpperLeg"),p("rightLowerLeg"),p("rightFoot"))
    },
    camera:{
      width:size.x,height:size.y,
      fov_deg:camera.fov,aspect:camera.aspect,near:camera.near,far:camera.far,
      projection_matrix:camera.projectionMatrix.toArray(),
      camera_to_world:cameraWorld.toArray(),
      world_to_camera:worldToCamera.toArray()
    },
    joints3d,
    joints2d
  };
}
