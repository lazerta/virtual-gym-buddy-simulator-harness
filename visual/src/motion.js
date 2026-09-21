import * as THREE from "three";
import {BIOMECH,FORM_MODIFIERS} from "./biomechanics.js";
import {resetRig,resetRoot,alignBoneToWorldPoint} from "./rig.js";
import {solveTwoBoneIK,distanceForFlexion,jointFlexionDeg,setWorldQuaternion} from "./ik.js";

const V=(x=0,y=0,z=0)=>new THREE.Vector3(x,y,z);
const wp=o=>o.getWorldPosition(new THREE.Vector3());
const lerp=(a,b,t)=>a.clone().lerp(b,t);
const clamp01=x=>Math.max(0,Math.min(1,x));
const smooth=x=>{x=clamp01(x);return x*x*(3-2*x)};
const sideScale=form=>form==="asymmetry"?FORM_MODIFIERS.asymmetry.sideScale:1;
const romScale=form=>form==="short_rom"?FORM_MODIFIERS.short_rom.romScale:1;

function setRootTilt(model,rootRest,deg){
  resetRoot(model,rootRest);
  model.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(V(1,0,0),THREE.MathUtils.degToRad(deg)));
  model.updateMatrixWorld(true);
}
function placeRootOnAnchor(model,rig,rootRest,anchor){
  resetRoot(model,rootRest);
  anchor.updateWorldMatrix(true,false);
  const aq=anchor.getWorldQuaternion(new THREE.Quaternion());
  model.quaternion.copy(aq.multiply(rootRest.quaternion.clone()));
  model.updateMatrixWorld(true);
  alignBoneToWorldPoint(model,rig.hips,anchor.getWorldPosition(new THREE.Vector3()));
}
function placeHips(model,rig,target){alignBoneToWorldPoint(model,rig.hips,target)}
function setControlWorldPosition(station,node,target){
  const p=station.group.worldToLocal(target.clone());
  node.position.copy(p);
}
function solveArm(rig,side,target,pole){
  return solveTwoBoneIK({
    upper:side==="L"?rig.leftUpperArm:rig.rightUpperArm,
    lower:side==="L"?rig.leftLowerArm:rig.rightLowerArm,
    end:side==="L"?rig.leftHand:rig.rightHand,
    target,pole
  });
}
function solveLeg(rig,side,target,pole){
  return solveTwoBoneIK({
    upper:side==="L"?rig.leftUpperLeg:rig.rightUpperLeg,
    lower:side==="L"?rig.leftLowerLeg:rig.rightLowerLeg,
    end:side==="L"?rig.leftFoot:rig.rightFoot,
    target,pole
  });
}
function mean2(a,b){return a.clone().add(b).multiplyScalar(.5)}
function rotateSpineToward(rig,leanDeg){
  const child=rig.chest||rig.neck;
  if(!rig.spine||!child)return;
  const start=wp(rig.spine), len=start.distanceTo(wp(child));
  const r=THREE.MathUtils.degToRad(leanDeg);
  const target=start.clone().add(V(0,Math.cos(r)*len,Math.sin(r)*len));
  const current=wp(child).sub(start).normalize(), desired=target.clone().sub(start).normalize();
  const delta=new THREE.Quaternion().setFromUnitVectors(current,desired);
  const worldQ=rig.spine.getWorldQuaternion(new THREE.Quaternion());
  const parentQ=rig.spine.parent.getWorldQuaternion(new THREE.Quaternion());
  rig.spine.quaternion.copy(parentQ.invert().multiply(delta.multiply(worldQ)));
  rig.spine.updateWorldMatrix(true,true);
}

function pressMotion(ctx){
  const {exercise,form,d,model,rig,rootRest,metrics,station,truth}=ctx;
  const spec=BIOMECH[exercise], rom=romScale(form);
  setRootTilt(model,rootRest,inclineDeg-90);
  placeHips(model,rig,wp(station.bodyAnchor));
  model.updateMatrixWorld(true);
  const lSh=wp(rig.leftUpperArm), rSh=wp(rig.rightUpperArm), sh=mean2(lSh,rSh);
  const grip=spec.gripShoulderRatio*metrics.shoulderWidth;
  const reach=metrics.armReach*.92;
  const bottomY=sh.y+.04, topY=sh.y+reach*.72;
  const y=THREE.MathUtils.lerp(bottomY,topY,d*rom);
  const z=THREE.MathUtils.lerp(sh.z-.10,sh.z+.06,d*rom);
  const left=V(sh.x-grip/2,y,z), right=V(sh.x+grip/2,y,z);
  const extra=form==="elbow_flare"?.12:0;
  const lp=mean2(lSh,left).add(V(-metrics.shoulderWidth*(.35+extra),-.10,.12));
  const rp=mean2(rSh,right).add(V(metrics.shoulderWidth*(.35+extra),-.10,.12));
  const e1=solveArm(rig,"L",left,lp), e2=solveArm(rig,"R",right,rp);
  if(station.bar){station.bar.position.y=y;station.bar.position.z=z}
  if(station.left){setControlWorldPosition(station,station.left,left);setControlWorldPosition(station,station.right,right)}
  if(form==="short_rom")truth.issues.rom_scale=rom;
  if(form==="elbow_flare")truth.issues.elbow_plane_modifier=20;
  if(form==="asymmetry")truth.issues.side_scale=sideScale(form);
  truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function squatMotion(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx,spec=BIOMECH.smith_squat;
  resetRoot(model,rootRest);resetRig(rig,ctx.rigRest);model.updateMatrixWorld(true);
  const footL=wp(rig.leftFoot),footR=wp(rig.rightFoot),footMid=mean2(footL,footR);
  const footLQ=rig.leftFoot.getWorldQuaternion(new THREE.Quaternion()),footRQ=rig.rightFoot.getWorldQuaternion(new THREE.Quaternion());
  const flex=spec.kneeFlexBottomDeg*d*romScale(form);
  const legDistance=distanceForFlexion(metrics.thigh,metrics.shin,flex);
  const zShift=.08*d;
  const vertical=Math.sqrt(Math.max(.01,legDistance*legDistance-zShift*zShift));
  const hipTarget=V(footMid.x,Math.min(wp(rig.hips).y,footMid.y+vertical),footMid.z+zShift);
  placeHips(model,rig,hipTarget);
  const inward=form==="knee_valgus"?.07*d:0;
  const leftPole=mean2(wp(rig.leftUpperLeg),footL).add(V(+inward,.05,.45));
  const rightPole=mean2(wp(rig.rightUpperLeg),footR).add(V(-inward,.05,.45));
  const le=solveTwoBoneIK({upper:rig.leftUpperLeg,lower:rig.leftLowerLeg,end:rig.leftFoot,target:footL,pole:leftPole,endWorldQuaternion:footLQ}), re=solveTwoBoneIK({upper:rig.rightUpperLeg,lower:rig.rightLowerLeg,end:rig.rightFoot,target:footR,pole:rightPole,endWorldQuaternion:footRQ});
  rotateSpineToward(rig,(form==="forward_lean"?spec.trunkLeanStressDeg:spec.trunkLeanNominalDeg)*d);
  model.updateMatrixWorld(true);
  if(station.bar){
    const shoulderMid=mean2(wp(rig.leftUpperArm),wp(rig.rightUpperArm));
    setControlWorldPosition(station,station.bar,shoulderMid);
    const grip=metrics.shoulderWidth*spec.gripShoulderRatio;
    const lt=V(shoulderMid.x-grip/2,shoulderMid.y,shoulderMid.z), rt=V(shoulderMid.x+grip/2,shoulderMid.y,shoulderMid.z);
    const lp=mean2(wp(rig.leftUpperArm),lt).add(V(-.2,0,.15)), rp=mean2(wp(rig.rightUpperArm),rt).add(V(.2,0,.15));
    solveArm(rig,"L",lt,lp);solveArm(rig,"R",rt,rp);
  }
  if(form==="short_rom")truth.issues.rom_scale=FORM_MODIFIERS.short_rom.romScale;
  if(form==="knee_valgus")truth.issues.knee_valgus_deg=FORM_MODIFIERS.knee_valgus.valgusDeg;
  if(form==="forward_lean")truth.issues.trunk_lean_deg=spec.trunkLeanStressDeg*d;
  truth.constraint_error_m=Math.max(le.error,re.error);
}

function ohpMotion(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx,spec=BIOMECH.seated_ohp,rom=romScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);model.updateMatrixWorld(true);
  const l=wp(rig.leftUpperArm),r=wp(rig.rightUpperArm),mid=mean2(l,r),grip=metrics.shoulderWidth*spec.gripShoulderRatio;
  const y=THREE.MathUtils.lerp(mid.y+.05,mid.y+metrics.armReach*.92,d*rom);
  const left=V(mid.x-grip/2,y,mid.z),right=V(mid.x+grip/2,y,mid.z);
  const e1=solveArm(rig,"L",left,mean2(l,left).add(V(-.25,0,.15))),e2=solveArm(rig,"R",right,mean2(r,right).add(V(.25,0,.15)));
  if(station.left){setControlWorldPosition(station,station.left,left);setControlWorldPosition(station,station.right,right)}
  if(form==="short_rom")truth.issues.rom_scale=rom;truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function lateralRaise(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx,rom=romScale(form),side=sideScale(form);
  resetRoot(model,rootRest);resetRig(rig,ctx.rigRest);model.updateMatrixWorld(true);
  const lSh=wp(rig.leftUpperArm),rSh=wp(rig.rightUpperArm),l0=wp(rig.leftHand),r0=wp(rig.rightHand);
  const lTop=V(lSh.x-metrics.armReach*.94,lSh.y,lSh.z),rTop=V(rSh.x+metrics.armReach*.94*side,rSh.y,rSh.z);
  const left=lerp(l0,lTop,d*rom),right=lerp(r0,rTop,d*rom);
  const e1=solveArm(rig,"L",left,mean2(lSh,left).add(V(-.1,.02,.18))),e2=solveArm(rig,"R",right,mean2(rSh,right).add(V(.1,.02,.18)));
  if(station.left){station.left.position.copy(station.group.worldToLocal(left.clone()));station.right.position.copy(station.group.worldToLocal(right.clone()))}
  if(form==="short_rom")truth.issues.rom_scale=rom;if(form==="asymmetry")truth.issues.side_scale=side;truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function latPulldown(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx,spec=BIOMECH.lat_pulldown,rom=romScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);model.updateMatrixWorld(true);
  const l=wp(rig.leftUpperArm),r=wp(rig.rightUpperArm),mid=mean2(l,r),grip=metrics.shoulderWidth*spec.gripShoulderRatio;
  const topY=mid.y+metrics.armReach*.95,bottomY=mid.y+.18;
  const y=THREE.MathUtils.lerp(topY,bottomY,d*rom),z=mid.z-.04;
  const left=V(mid.x-grip/2,y,z),right=V(mid.x+grip/2,y,z);
  const e1=solveArm(rig,"L",left,mean2(l,left).add(V(-.30,0,.15))),e2=solveArm(rig,"R",right,mean2(r,right).add(V(.30,0,.15)));
  setControlWorldPosition(station,station.bar,V(mid.x,y,z));
  if(form==="forward_lean"){rotateSpineToward(rig,18*d);truth.issues.trunk_lean_deg=18*d}
  if(form==="short_rom")truth.issues.rom_scale=rom;truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function tRow(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx,spec=BIOMECH.chest_supported_t_row,rom=romScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);model.updateMatrixWorld(true);
  const l=wp(rig.leftUpperArm),r=wp(rig.rightUpperArm),mid=mean2(l,r),far=mid.z-.75,near=mid.z-.22;
  const z=THREE.MathUtils.lerp(far,near,d*rom),y=mid.y-.15,half=metrics.shoulderWidth*.42;
  const left=V(mid.x-half,y,z),right=V(mid.x+half,y,z);
  const flare=form==="elbow_flare"?.18:.06;
  const e1=solveArm(rig,"L",left,mean2(l,left).add(V(-flare,0,.25))),e2=solveArm(rig,"R",right,mean2(r,right).add(V(flare,0,.25)));
  setControlWorldPosition(station,station.handle,V(mid.x,y,z));
  if(form==="elbow_flare")truth.issues.elbow_plane_modifier=20;if(form==="short_rom")truth.issues.rom_scale=rom;truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function legPress(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx,spec=BIOMECH.leg_press,rom=romScale(form),side=sideScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);model.updateMatrixWorld(true);
  const flex=THREE.MathUtils.lerp(spec.kneeFlexTopDeg,spec.kneeFlexBottomDeg,d*rom);
  const reach=distanceForFlexion(metrics.thigh,metrics.shin,flex),rail=V(0,Math.sin(Math.PI/4),-Math.cos(Math.PI/4));
  const lHip=wp(rig.leftUpperLeg),rHip=wp(rig.rightUpperLeg);
  const left=lHip.clone().add(rail.clone().multiplyScalar(reach)),right=rHip.clone().add(rail.clone().multiplyScalar(reach*side));
  const e1=solveLeg(rig,"L",left,mean2(lHip,left).add(V(-.05,.15,.25))),e2=solveLeg(rig,"R",right,mean2(rHip,right).add(V(.05,.15,.25)));
  const center=mean2(left,right);station.sled.position.copy(station.group.worldToLocal(center.clone()));
  if(form==="knee_valgus")truth.issues.knee_valgus_deg=8;if(form==="short_rom")truth.issues.rom_scale=rom;if(form==="asymmetry")truth.issues.side_scale=side;truth.constraint_error_m=Math.max(e1.error,e2.error);
}

export function applyMotion(ctx){
  resetRig(ctx.rig,ctx.rigRest);resetRoot(ctx.model,ctx.rootRest);ctx.model.updateMatrixWorld(true);
  const d=smooth(ctx.phase);ctx={...ctx,d,truth:{exercise:ctx.exercise,form:ctx.form,phase:d,issues:{},constraint_error_m:0}};
  if(ctx.exercise==="smith_squat")squatMotion(ctx);
  else if(["flat_db_press","barbell_bench"].includes(ctx.exercise))pressMotion(ctx,0);
  else if(["incline_db_press","incline_smith_press"].includes(ctx.exercise))pressMotion(ctx,BIOMECH[ctx.exercise].benchDeg);
  else if(ctx.exercise==="seated_ohp")ohpMotion(ctx);
  else if(ctx.exercise==="leg_press")legPress(ctx);
  else if(ctx.exercise==="lateral_raise")lateralRaise(ctx);
  else if(ctx.exercise==="lat_pulldown")latPulldown(ctx);
  else if(ctx.exercise==="chest_supported_t_row")tRow(ctx);
  ctx.model.updateMatrixWorld(true);
  const p=n=>ctx.rig[n]?wp(ctx.rig[n]):null;
  ctx.truth.measured={
    left_elbow_flex_deg:jointFlexionDeg(p("leftUpperArm"),p("leftLowerArm"),p("leftHand")),
    right_elbow_flex_deg:jointFlexionDeg(p("rightUpperArm"),p("rightLowerArm"),p("rightHand")),
    left_knee_flex_deg:jointFlexionDeg(p("leftUpperLeg"),p("leftLowerLeg"),p("leftFoot")),
    right_knee_flex_deg:jointFlexionDeg(p("rightUpperLeg"),p("rightLowerLeg"),p("rightFoot"))
  };
  return ctx.truth;
}
export function cyclePhase(t){return (Math.sin(t*1.6-Math.PI/2)+1)/2}
