export function findBone(root, candidates){
  let found=null;
  root.traverse(o=>{
    if(found||!o.isBone) return;
    const n=o.name.toLowerCase().replace(/[^a-z0-9]/g,"");
    if(candidates.some(c=>n.includes(c))) found=o;
  });
  return found;
}

export function resolveRig(root){
  const map={
    hips:["hips","pelvis"],
    spine:["spine1","spine2","spine","chest"],
    leftUpperLeg:["leftupleg","leftthigh","leftupperleg"],
    rightUpperLeg:["rightupleg","rightthigh","rightupperleg"],
    leftLowerLeg:["leftleg","leftcalf","leftlowerleg"],
    rightLowerLeg:["rightleg","rightcalf","rightlowerleg"],
    leftUpperArm:["leftarm","leftupperarm"],
    rightUpperArm:["rightarm","rightupperarm"],
    leftLowerArm:["leftforearm","leftlowerarm"],
    rightLowerArm:["rightforearm","rightlowerarm"]
  };
  const out={};
  for(const [k,v] of Object.entries(map)) out[k]=findBone(root,v);
  return out;
}

export function captureRest(rig){
  const rest={};
  for(const [k,b] of Object.entries(rig)) if(b) rest[k]=b.quaternion.clone();
  return rest;
}

export function resetRig(rig,rest){
  for(const [k,b] of Object.entries(rig)) if(b&&rest[k]) b.quaternion.copy(rest[k]);
}
