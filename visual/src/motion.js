import * as THREE from "three";

const clamp01=x=>Math.max(0,Math.min(1,x));
const smooth=x=>{x=clamp01(x);return x*x*(3-2*x)};
const rot=(bone,axis,deg)=>{if(!bone)return; const q=new THREE.Quaternion().setFromAxisAngle(axis,THREE.MathUtils.degToRad(deg)); bone.quaternion.multiply(q)};

export function applyMotion({exercise,form,phase,model,rig,rest,equipment}){
  for(const [k,b] of Object.entries(rig)) if(b&&rest[k]) b.quaternion.copy(rest[k]);
  const d=smooth(phase);
  model.position.y=0;
  let truth={exercise,form,phase:d,issues:{}};

  if(exercise==="smith_squat"){
    let depth=form==="short_rom"?d*.55:d;
    let lean=(form==="forward_lean"?28:12)*depth;
    model.position.y=-.42*depth;
    rot(rig.leftUpperLeg,new THREE.Vector3(1,0,0),55*depth);
    rot(rig.rightUpperLeg,new THREE.Vector3(1,0,0),55*depth);
    rot(rig.leftLowerLeg,new THREE.Vector3(1,0,0),-78*depth);
    rot(rig.rightLowerLeg,new THREE.Vector3(1,0,0),-78*depth);
    rot(rig.spine,new THREE.Vector3(1,0,0),lean);
    if(form==="knee_valgus"){
      rot(rig.leftUpperLeg,new THREE.Vector3(0,0,1),9*depth);
      rot(rig.rightUpperLeg,new THREE.Vector3(0,0,1),-9*depth);
      truth.issues.knee_valgus_deg=9*depth;
    }
    if(form==="short_rom") truth.issues.rom_scale=.55;
    if(form==="forward_lean") truth.issues.trunk_lean_deg=lean;
    if(equipment?.bar) equipment.bar.position.y=1.8-.42*depth;
  }

  if(exercise==="incline_smith_press"){
    const press=1-d;
    model.position.set(0,0,.15);
    rot(rig.leftUpperArm,new THREE.Vector3(1,0,0),-55);
    rot(rig.rightUpperArm,new THREE.Vector3(1,0,0),-55);
    rot(rig.leftLowerArm,new THREE.Vector3(1,0,0),65*d);
    rot(rig.rightLowerArm,new THREE.Vector3(1,0,0),65*d);
    if(form==="elbow_flare"){
      rot(rig.leftUpperArm,new THREE.Vector3(0,0,1),18);
      rot(rig.rightUpperArm,new THREE.Vector3(0,0,1),-18);
      truth.issues.elbow_flare_deg=18;
    }
    if(equipment?.bar) equipment.bar.position.y=1.35+.45*press;
  }

  if(exercise==="lateral_raise"){
    const angle=85*d;
    rot(rig.leftUpperArm,new THREE.Vector3(0,0,1),angle);
    rot(rig.rightUpperArm,new THREE.Vector3(0,0,1),-angle);
    if(form==="short_rom") truth.issues.rom_scale=.6;
  }
  return truth;
}

export function cyclePhase(t){
  return (Math.sin(t*1.6-Math.PI/2)+1)/2;
}
