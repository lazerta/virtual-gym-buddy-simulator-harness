import * as THREE from "three";

const clamp01=x=>Math.max(0,Math.min(1,x));
const smooth=x=>{x=clamp01(x);return x*x*(3-2*x)};
const AX={x:new THREE.Vector3(1,0,0),y:new THREE.Vector3(0,1,0),z:new THREE.Vector3(0,0,1)};
const rot=(bone,axis,deg)=>{if(!bone)return; bone.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(axis,THREE.MathUtils.degToRad(deg)))};
const q=(form,name,a,b)=>form===name?b:a;

export function applyMotion({exercise,form,phase,model,rig,rest,equipment}){
  for(const [k,b] of Object.entries(rig)) if(b&&rest[k]) b.quaternion.copy(rest[k]);
  model.position.set(0,0,0.15);
  const d=smooth(phase), truth={exercise,form,phase:d,issues:{}};
  const asym=form==="asymmetry"?.82:1;

  if(exercise==="smith_squat"){
    const depth=q(form,"short_rom",d,d*.55), lean=q(form,"forward_lean",12,28)*depth;
    model.position.y=.15-.42*depth;
    rot(rig.leftUpperLeg,AX.x,55*depth); rot(rig.rightUpperLeg,AX.x,55*depth*asym);
    rot(rig.leftLowerLeg,AX.x,-78*depth); rot(rig.rightLowerLeg,AX.x,-78*depth*asym); rot(rig.spine,AX.x,lean);
    if(form==="knee_valgus"){rot(rig.leftUpperLeg,AX.z,9*depth);rot(rig.rightUpperLeg,AX.z,-9*depth);truth.issues.knee_valgus_deg=9*depth}
    if(form==="short_rom")truth.issues.rom_scale=.55;if(form==="forward_lean")truth.issues.trunk_lean_deg=lean;if(form==="asymmetry")truth.issues.side_scale=.82;
    if(equipment?.bar)equipment.bar.position.y=1.8-.42*depth;
  }

  if(["incline_db_press","flat_db_press","barbell_bench","incline_smith_press"].includes(exercise)){
    const incline=["incline_db_press","incline_smith_press"].includes(exercise)?32:0;
    const rom=q(form,"short_rom",1,.58), flex=65*d*rom, flare=q(form,"elbow_flare",8,22);
    model.rotation.x=THREE.MathUtils.degToRad(-incline*.55); model.position.y=.48;
    rot(rig.leftUpperArm,AX.x,-58);rot(rig.rightUpperArm,AX.x,-58*asym);rot(rig.leftUpperArm,AX.z,flare);rot(rig.rightUpperArm,AX.z,-flare);
    rot(rig.leftLowerArm,AX.x,flex);rot(rig.rightLowerArm,AX.x,flex*asym);
    if(form==="elbow_flare")truth.issues.elbow_flare_deg=22;if(form==="short_rom")truth.issues.rom_scale=.58;if(form==="asymmetry")truth.issues.side_scale=.82;
    if(equipment?.bar)equipment.bar.position.y=(incline?1.45:1.15)+.42*(1-d*rom);
    if(equipment?.left){equipment.left.position.y=1.0+.35*(1-d*rom);equipment.right.position.y=1.0+.35*(1-d*rom)}
  }

  if(exercise==="seated_ohp"){
    const rom=q(form,"short_rom",1,.6), flex=78*d*rom;
    model.position.y=.12;rot(rig.leftUpperArm,AX.x,-20);rot(rig.rightUpperArm,AX.x,-20*asym);rot(rig.leftUpperArm,AX.z,48);rot(rig.rightUpperArm,AX.z,-48);
    rot(rig.leftLowerArm,AX.x,flex);rot(rig.rightLowerArm,AX.x,flex*asym);
    if(form==="elbow_flare")truth.issues.elbow_flare_deg=20;if(form==="short_rom")truth.issues.rom_scale=.6;
    if(equipment?.left){equipment.left.rotation.z=-.35*d;equipment.right.rotation.z=.35*d}
  }

  if(exercise==="leg_press"){
    const rom=q(form,"short_rom",1,.58), depth=d*rom;
    model.rotation.x=THREE.MathUtils.degToRad(-42);model.position.y=.42;model.position.z=.35;
    rot(rig.leftUpperLeg,AX.x,68*depth);rot(rig.rightUpperLeg,AX.x,68*depth*asym);rot(rig.leftLowerLeg,AX.x,-88*depth);rot(rig.rightLowerLeg,AX.x,-88*depth*asym);
    if(form==="knee_valgus"){rot(rig.leftUpperLeg,AX.z,8*depth);rot(rig.rightUpperLeg,AX.z,-8*depth);truth.issues.knee_valgus_deg=8*depth}
    if(form==="short_rom")truth.issues.rom_scale=.58;
    if(equipment?.sled)equipment.sled.position.z=-.05-.35*(1-depth);
  }

  if(exercise==="lateral_raise"){
    const rom=q(form,"short_rom",1,.6), angle=85*d*rom;
    rot(rig.leftUpperArm,AX.z,angle);rot(rig.rightUpperArm,AX.z,-angle*asym);
    if(form==="short_rom")truth.issues.rom_scale=.6;if(form==="asymmetry")truth.issues.side_scale=.82;
    if(equipment?.left){equipment.left.position.y=1.0+.55*d*rom;equipment.right.position.y=1.0+.55*d*rom*asym}
  }

  if(exercise==="lat_pulldown"){
    const rom=q(form,"short_rom",1,.6), pull=d*rom; model.position.y=.1;
    rot(rig.leftUpperArm,AX.z,150-75*pull);rot(rig.rightUpperArm,AX.z,-(150-75*pull)*asym);rot(rig.leftLowerArm,AX.x,50*pull);rot(rig.rightLowerArm,AX.x,50*pull*asym);
    if(form==="forward_lean"){rot(rig.spine,AX.x,-18*pull);truth.issues.trunk_lean_deg=18*pull}
    if(form==="short_rom")truth.issues.rom_scale=.6;if(equipment?.bar)equipment.bar.position.y=2.05-.55*pull;
  }

  if(exercise==="chest_supported_t_row"){
    const rom=q(form,"short_rom",1,.6), pull=d*rom; model.rotation.x=THREE.MathUtils.degToRad(-20); model.position.y=.35;
    rot(rig.leftUpperArm,AX.x,-25-45*pull);rot(rig.rightUpperArm,AX.x,(-25-45*pull)*asym);rot(rig.leftLowerArm,AX.x,75*pull);rot(rig.rightLowerArm,AX.x,75*pull*asym);
    if(form==="elbow_flare"){rot(rig.leftUpperArm,AX.z,18);rot(rig.rightUpperArm,AX.z,-18);truth.issues.elbow_flare_deg=18}
    if(form==="short_rom")truth.issues.rom_scale=.6;if(equipment?.handle)equipment.handle.position.z=-.65+.35*pull;
  }

  return truth;
}

export function cyclePhase(t){return (Math.sin(t*1.6-Math.PI/2)+1)/2}
