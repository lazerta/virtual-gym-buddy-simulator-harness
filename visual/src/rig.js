import * as THREE from "three";

const norm=s=>s.toLowerCase().replace(/[^a-z0-9]/g,"");

export function findBone(root,candidates){
  const wanted=candidates.map(norm);
  let exact=null, partial=null;
  root.traverse(o=>{
    if(!o.isBone) return;
    const n=norm(o.name);
    if(wanted.includes(n)) exact=o;
    else if(!partial && wanted.some(c=>n.endsWith(c)||n.includes(c))) partial=o;
  });
  return exact||partial;
}

export function resolveRig(root){
  const map={
    hips:["hips","pelvis"], spine:["spine","spine1"], chest:["spine2","chest","upperchest"],
    neck:["neck"], head:["head"],
    leftShoulder:["leftshoulder"], rightShoulder:["rightshoulder"],
    leftUpperArm:["leftarm","leftupperarm"], rightUpperArm:["rightarm","rightupperarm"],
    leftLowerArm:["leftforearm","leftlowerarm"], rightLowerArm:["rightforearm","rightlowerarm"],
    leftHand:["lefthand"], rightHand:["righthand"],
    leftUpperLeg:["leftupleg","leftthigh","leftupperleg"], rightUpperLeg:["rightupleg","rightthigh","rightupperleg"],
    leftLowerLeg:["leftleg","leftcalf","leftlowerleg"], rightLowerLeg:["rightleg","rightcalf","rightlowerleg"],
    leftFoot:["leftfoot","leftankle"], rightFoot:["rightfoot","rightankle"]
  };
  const out={}; for(const [k,v] of Object.entries(map)) out[k]=findBone(root,v); return out;
}

export function captureRest(rig){
  const rest={}; for(const [k,b] of Object.entries(rig)) if(b) rest[k]=b.quaternion.clone(); return rest;
}
export function captureRootRest(model){return {position:model.position.clone(),quaternion:model.quaternion.clone(),scale:model.scale.clone()}}
export function resetRig(rig,rest){for(const [k,b] of Object.entries(rig)) if(b&&rest[k]) b.quaternion.copy(rest[k])}
export function resetRoot(model,rest){model.position.copy(rest.position);model.quaternion.copy(rest.quaternion);model.scale.copy(rest.scale)}

export function normalizeAvatar(model,targetHeight=1.75){
  model.updateMatrixWorld(true);
  let box=new THREE.Box3().setFromObject(model), size=box.getSize(new THREE.Vector3());
  if(size.y<=1e-6) throw new Error("avatar has zero height");
  const s=targetHeight/size.y; model.scale.multiplyScalar(s); model.updateMatrixWorld(true);
  box=new THREE.Box3().setFromObject(model); model.position.y-=box.min.y; model.updateMatrixWorld(true);
  return s;
}

const wp=b=>b?.getWorldPosition(new THREE.Vector3())??null;
const dist=(a,b)=>a&&b?a.distanceTo(b):0;
export function measureRig(model,rig){
  model.updateMatrixWorld(true);
  const p={}; for(const [k,b] of Object.entries(rig)) if(b) p[k]=wp(b);
  const box=new THREE.Box3().setFromObject(model), size=box.getSize(new THREE.Vector3());
  return {
    height:size.y,
    upperArm:(dist(p.leftUpperArm,p.leftLowerArm)+dist(p.rightUpperArm,p.rightLowerArm))/2,
    forearm:(dist(p.leftLowerArm,p.leftHand)+dist(p.rightLowerArm,p.rightHand))/2,
    thigh:(dist(p.leftUpperLeg,p.leftLowerLeg)+dist(p.rightUpperLeg,p.rightLowerLeg))/2,
    shin:(dist(p.leftLowerLeg,p.leftFoot)+dist(p.rightLowerLeg,p.rightFoot))/2,
    shoulderWidth:dist(p.leftUpperArm,p.rightUpperArm),
    hipWidth:dist(p.leftUpperLeg,p.rightUpperLeg),
    armReach:(dist(p.leftUpperArm,p.leftLowerArm)+dist(p.leftLowerArm,p.leftHand)+dist(p.rightUpperArm,p.rightLowerArm)+dist(p.rightLowerArm,p.rightHand))/2,
    legReach:(dist(p.leftUpperLeg,p.leftLowerLeg)+dist(p.leftLowerLeg,p.leftFoot)+dist(p.rightUpperLeg,p.rightLowerLeg)+dist(p.rightLowerLeg,p.rightFoot))/2,
    restPositions:p
  };
}

export function alignBoneToWorldPoint(model,bone,target){
  model.updateMatrixWorld(true); const cur=bone.getWorldPosition(new THREE.Vector3()); model.position.add(target.clone().sub(cur)); model.updateMatrixWorld(true);
}
