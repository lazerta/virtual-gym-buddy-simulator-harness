import * as THREE from "three";
import {BIOMECH,FORM_MODIFIERS} from "./biomechanics.js";
import {resetRig,resetRoot,alignBoneToWorldPoint} from "./rig.js";
import {solveTwoBoneIK,distanceForFlexion,jointFlexionDeg} from "./ik.js";

const V=(x=0,y=0,z=0)=>new THREE.Vector3(x,y,z);
const wp=o=>o.getWorldPosition(new THREE.Vector3());
const wq=o=>o.getWorldQuaternion(new THREE.Quaternion());
const mean2=(a,b)=>a.clone().add(b).multiplyScalar(.5);
const lerp=(a,b,t)=>a.clone().lerp(b,t);
const clamp01=x=>Math.max(0,Math.min(1,x));
const smooth=x=>{x=clamp01(x);return x*x*(3-2*x)};
const sideScale=form=>form==="asymmetry"?FORM_MODIFIERS.asymmetry.sideScale:1;
const romScale=form=>form==="short_rom"?FORM_MODIFIERS.short_rom.romScale:1;

function placeRootOnAnchor(model,rig,rootRest,anchor){
  resetRoot(model,rootRest);
  anchor.updateWorldMatrix(true,false);
  const anchorQ=anchor.getWorldQuaternion(new THREE.Quaternion());
  model.quaternion.copy(anchorQ.multiply(rootRest.quaternion.clone()));
  model.updateMatrixWorld(true);
  alignBoneToWorldPoint(model,rig.hips,anchor.getWorldPosition(new THREE.Vector3()));
}
function placeHips(model,rig,target){alignBoneToWorldPoint(model,rig.hips,target)}
function setControlWorldPosition(station,node,target){
  station.group.updateWorldMatrix(true,false);
  node.position.copy(station.group.worldToLocal(target.clone()));
}
function solveArm(rig,side,target,pole){
  return solveTwoBoneIK({
    upper:side==="L"?rig.leftUpperArm:rig.rightUpperArm,
    lower:side==="L"?rig.leftLowerArm:rig.rightLowerArm,
    end:side==="L"?rig.leftHand:rig.rightHand,
    target,pole
  });
}
function solveLeg(rig,side,target,pole,endWorldQuaternion=null){
  return solveTwoBoneIK({
    upper:side==="L"?rig.leftUpperLeg:rig.rightUpperLeg,
    lower:side==="L"?rig.leftLowerLeg:rig.rightLowerLeg,
    end:side==="L"?rig.leftFoot:rig.rightFoot,
    target,pole,endWorldQuaternion
  });
}
function rotateSpineToward(rig,leanDeg){
  const child=rig.chest||rig.neck;
  if(!rig.spine||!child)return;
  const start=wp(rig.spine),len=start.distanceTo(wp(child));
  const r=THREE.MathUtils.degToRad(leanDeg);
  const target=start.clone().add(V(0,Math.cos(r)*len,Math.sin(r)*len));
  const current=wp(child).sub(start).normalize(),desired=target.clone().sub(start).normalize();
  const delta=new THREE.Quaternion().setFromUnitVectors(current,desired);
  const desiredWorld=delta.multiply(wq(rig.spine));
  const parentQ=wq(rig.spine.parent).invert();
  rig.spine.quaternion.copy(parentQ.multiply(desiredWorld));
  rig.spine.updateWorldMatrix(true,true);
}

function pressMotion(ctx){
  const {exercise,form,d,model,rig,rootRest,metrics,station,truth}=ctx;
  const spec=BIOMECH[exercise],rom=romScale(form),side=sideScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);
  model.updateMatrixWorld(true);

  const lSh=wp(rig.leftUpperArm),rSh=wp(rig.rightUpperArm),sh=mean2(lSh,rSh);
  const grip=spec.gripShoulderRatio*metrics.shoulderWidth;
  const reach=metrics.armReach*.92;
  const bottom=V(sh.x,sh.y+.04,sh.z-.10);
  const top=V(sh.x,sh.y+reach*.72,sh.z+.06);
  const center=lerp(bottom,top,d*rom);

  const left=V(center.x-grip/2,center.y,center.z);
  const right=V(center.x+grip/2,THREE.MathUtils.lerp(bottom.y,top.y,d*rom*side),center.z);
  const flare=form==="elbow_flare"?.47:.35;
  const lp=mean2(lSh,left).add(V(-metrics.shoulderWidth*flare,-.10,.12));
  const rp=mean2(rSh,right).add(V(metrics.shoulderWidth*flare,-.10,.12));

  const e1=solveArm(rig,"L",left,lp),e2=solveArm(rig,"R",right,rp);
  if(station.bar)setControlWorldPosition(station,station.bar,center);
  if(station.left){
    setControlWorldPosition(station,station.left,left);
    setControlWorldPosition(station,station.right,right);
  }

  if(form==="short_rom")truth.issues.rom_scale=rom;
  if(form==="elbow_flare")truth.issues.elbow_plane_modifier_deg=20;
  if(form==="asymmetry")truth.issues.side_scale=side;
  truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function squatMotion(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx;
  const spec=BIOMECH.smith_squat,rom=romScale(form);
  resetRoot(model,rootRest);resetRig(rig,ctx.rigRest);model.updateMatrixWorld(true);

  const footL=wp(rig.leftFoot),footR=wp(rig.rightFoot),footMid=mean2(footL,footR);
  const footLQ=wq(rig.leftFoot),footRQ=wq(rig.rightFoot);
  const prior=ctx.motionPriors?.squat?.sample(d*rom)??null;
  const flex=prior?.kneeFlexDeg??spec.kneeFlexBottomDeg*d*rom;
  const legDistance=distanceForFlexion(metrics.thigh,metrics.shin,flex);
  const zShift=prior?THREE.MathUtils.clamp(prior.hipForwardNorm*metrics.legReach*.12,-.12,.18):.08*d;
  const vertical=Math.sqrt(Math.max(.01,legDistance*legDistance-zShift*zShift));
  const hipTarget=V(footMid.x,Math.min(wp(rig.hips).y,footMid.y+vertical),footMid.z+zShift);
  placeHips(model,rig,hipTarget);

  const inward=form==="knee_valgus"?.07*d:0;
  const leftPole=mean2(wp(rig.leftUpperLeg),footL).add(V(+inward,.05,.45));
  const rightPole=mean2(wp(rig.rightUpperLeg),footR).add(V(-inward,.05,.45));
  const le=solveLeg(rig,"L",footL,leftPole,footLQ);
  const re=solveLeg(rig,"R",footR,rightPole,footRQ);

  const priorLean=prior?THREE.MathUtils.clamp(prior.trunkLeanDeg,0,40):spec.trunkLeanNominalDeg*d;
  rotateSpineToward(rig,form==="forward_lean"?spec.trunkLeanStressDeg*d:priorLean);
  model.updateMatrixWorld(true);

  if(station.bar){
    const shoulderMid=mean2(wp(rig.leftUpperArm),wp(rig.rightUpperArm));
    setControlWorldPosition(station,station.bar,shoulderMid);
    const grip=metrics.shoulderWidth*spec.gripShoulderRatio;
    const lt=V(shoulderMid.x-grip/2,shoulderMid.y,shoulderMid.z);
    const rt=V(shoulderMid.x+grip/2,shoulderMid.y,shoulderMid.z);
    solveArm(rig,"L",lt,mean2(wp(rig.leftUpperArm),lt).add(V(-.2,0,.15)));
    solveArm(rig,"R",rt,mean2(wp(rig.rightUpperArm),rt).add(V(.2,0,.15)));
  }

  if(form==="short_rom")truth.issues.rom_scale=rom;
  if(form==="knee_valgus")truth.issues.knee_valgus_deg=FORM_MODIFIERS.knee_valgus.valgusDeg;
  if(form==="forward_lean")truth.issues.trunk_lean_deg=spec.trunkLeanStressDeg*d;
  truth.motion_prior=prior?{source:ctx.motionPriors.squat.source,knee_flex_deg:flex,trunk_lean_deg:priorLean}:null;
  truth.constraint_error_m=Math.max(le.error,re.error);
}

function ohpMotion(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx;
  const spec=BIOMECH.seated_ohp,rom=romScale(form),side=sideScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);
  model.updateMatrixWorld(true);

  const l=wp(rig.leftUpperArm),r=wp(rig.rightUpperArm),mid=mean2(l,r);
  const grip=metrics.shoulderWidth*spec.gripShoulderRatio;
  const bottom=V(mid.x,mid.y+.05,mid.z);
  const top=V(mid.x,mid.y+metrics.armReach*.92,mid.z);
  const center=lerp(bottom,top,d*rom);
  const left=V(center.x-grip/2,center.y,center.z);
  const right=V(center.x+grip/2,THREE.MathUtils.lerp(bottom.y,top.y,d*rom*side),center.z);
  const e1=solveArm(rig,"L",left,mean2(l,left).add(V(-.25,0,.15)));
  const e2=solveArm(rig,"R",right,mean2(r,right).add(V(.25,0,.15)));

  if(station.left){
    setControlWorldPosition(station,station.left,left);
    setControlWorldPosition(station,station.right,right);
  }
  if(form==="short_rom")truth.issues.rom_scale=rom;
  if(form==="asymmetry")truth.issues.side_scale=side;
  truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function lateralRaise(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx;
  const rom=romScale(form),side=sideScale(form);
  resetRoot(model,rootRest);resetRig(rig,ctx.rigRest);model.updateMatrixWorld(true);

  const lSh=wp(rig.leftUpperArm),rSh=wp(rig.rightUpperArm);
  const l0=wp(rig.leftHand),r0=wp(rig.rightHand);
  const lTop=V(lSh.x-metrics.armReach*.94,lSh.y,lSh.z);
  const rTop=V(rSh.x+metrics.armReach*.94,rSh.y,rSh.z);
  const left=lerp(l0,lTop,d*rom);
  const right=lerp(r0,rTop,d*rom*side);
  const e1=solveArm(rig,"L",left,mean2(lSh,left).add(V(-.10,.02,.18)));
  const e2=solveArm(rig,"R",right,mean2(rSh,right).add(V(.10,.02,.18)));

  setControlWorldPosition(station,station.left,left);
  setControlWorldPosition(station,station.right,right);
  if(form==="short_rom")truth.issues.rom_scale=rom;
  if(form==="asymmetry")truth.issues.side_scale=side;
  truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function latPulldown(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx;
  const spec=BIOMECH.lat_pulldown,rom=romScale(form),side=sideScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);
  model.updateMatrixWorld(true);

  const l=wp(rig.leftUpperArm),r=wp(rig.rightUpperArm),mid=mean2(l,r);
  const grip=metrics.shoulderWidth*spec.gripShoulderRatio;
  const topY=mid.y+metrics.armReach*.95,bottomY=mid.y+.18;
  const leftY=THREE.MathUtils.lerp(topY,bottomY,d*rom);
  const rightY=THREE.MathUtils.lerp(topY,bottomY,d*rom*side);
  const left=V(mid.x-grip/2,leftY,mid.z-.04);
  const right=V(mid.x+grip/2,rightY,mid.z-.04);
  const e1=solveArm(rig,"L",left,mean2(l,left).add(V(-.30,0,.15)));
  const e2=solveArm(rig,"R",right,mean2(r,right).add(V(.30,0,.15)));

  setControlWorldPosition(station,station.bar,mean2(left,right));
  if(form==="forward_lean"){
    rotateSpineToward(rig,18*d);
    truth.issues.trunk_lean_deg=18*d;
  }
  if(form==="short_rom")truth.issues.rom_scale=rom;
  if(form==="asymmetry")truth.issues.side_scale=side;
  truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function tRow(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx;
  const rom=romScale(form),side=sideScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);
  model.updateMatrixWorld(true);

  const l=wp(rig.leftUpperArm),r=wp(rig.rightUpperArm),mid=mean2(l,r);
  const far=mid.z-.75,near=mid.z-.22,y=mid.y-.15,half=metrics.shoulderWidth*.42;
  const left=V(mid.x-half,y,THREE.MathUtils.lerp(far,near,d*rom));
  const right=V(mid.x+half,y,THREE.MathUtils.lerp(far,near,d*rom*side));
  const flare=form==="elbow_flare"?.18:.06;
  const e1=solveArm(rig,"L",left,mean2(l,left).add(V(-flare,0,.25)));
  const e2=solveArm(rig,"R",right,mean2(r,right).add(V(flare,0,.25)));

  setControlWorldPosition(station,station.handle,mean2(left,right));
  if(form==="elbow_flare")truth.issues.elbow_plane_modifier_deg=20;
  if(form==="short_rom")truth.issues.rom_scale=rom;
  if(form==="asymmetry")truth.issues.side_scale=side;
  truth.constraint_error_m=Math.max(e1.error,e2.error);
}

function legPress(ctx){
  const {form,d,model,rig,rootRest,metrics,station,truth}=ctx;
  const spec=BIOMECH.leg_press,rom=romScale(form),side=sideScale(form);
  placeRootOnAnchor(model,rig,rootRest,station.bodyAnchor);
  model.updateMatrixWorld(true);

  const flexL=THREE.MathUtils.lerp(spec.kneeFlexTopDeg,spec.kneeFlexBottomDeg,d*rom);
  const flexR=THREE.MathUtils.lerp(spec.kneeFlexTopDeg,spec.kneeFlexBottomDeg,d*rom*side);
  const reachL=distanceForFlexion(metrics.thigh,metrics.shin,flexL);
  const reachR=distanceForFlexion(metrics.thigh,metrics.shin,flexR);
  const rail=V(0,Math.sin(Math.PI/4),-Math.cos(Math.PI/4));
  const lHip=wp(rig.leftUpperLeg),rHip=wp(rig.rightUpperLeg);
  const left=lHip.clone().add(rail.clone().multiplyScalar(reachL));
  const right=rHip.clone().add(rail.clone().multiplyScalar(reachR));
  const e1=solveLeg(rig,"L",left,mean2(lHip,left).add(V(-.05,.15,.25)));
  const e2=solveLeg(rig,"R",right,mean2(rHip,right).add(V(.05,.15,.25)));

  setControlWorldPosition(station,station.sled,mean2(left,right));
  if(form==="knee_valgus")truth.issues.knee_valgus_deg=8;
  if(form==="short_rom")truth.issues.rom_scale=rom;
  if(form==="asymmetry")truth.issues.side_scale=side;
  truth.constraint_error_m=Math.max(e1.error,e2.error);
}

export function applyMotion(input){
  resetRig(input.rig,input.rigRest);
  resetRoot(input.model,input.rootRest);
  input.model.updateMatrixWorld(true);

  const d=smooth(input.phase);
  const ctx={...input,d,truth:{exercise:input.exercise,form:input.form,phase:d,issues:{},constraint_error_m:0}};

  if(ctx.exercise==="smith_squat")squatMotion(ctx);
  else if(["flat_db_press","barbell_bench","incline_db_press","incline_smith_press"].includes(ctx.exercise))pressMotion(ctx);
  else if(ctx.exercise==="seated_ohp")ohpMotion(ctx);
  else if(ctx.exercise==="leg_press")legPress(ctx);
  else if(ctx.exercise==="lateral_raise")lateralRaise(ctx);
  else if(ctx.exercise==="lat_pulldown")latPulldown(ctx);
  else if(ctx.exercise==="chest_supported_t_row")tRow(ctx);
  else throw new Error(`unsupported exercise: ${ctx.exercise}`);

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

export function cyclePhase(t){return(Math.sin(t*1.6-Math.PI/2)+1)/2}
