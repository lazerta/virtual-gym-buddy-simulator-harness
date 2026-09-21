import * as THREE from "three";
import {BVHLoader} from "three/addons/loaders/BVHLoader.js";
import {jointFlexionDeg} from "./ik.js";

const norm=s=>s.toLowerCase().replace(/[^a-z0-9]/g,"");
const wp=b=>b.getWorldPosition(new THREE.Vector3());

function findBone(skeleton,candidates){
  const wanted=candidates.map(norm);
  return skeleton.bones.find(b=>wanted.includes(norm(b.name)))
    ||skeleton.bones.find(b=>wanted.some(x=>norm(b.name).includes(x)))
    ||null;
}
function lerp(a,b,t){return a+(b-a)*t}
function interp(samples,x,key){
  if(!samples.length)return null;
  const u=Math.max(0,Math.min(1,x))*(samples.length-1);
  const i=Math.floor(u),j=Math.min(samples.length-1,i+1),t=u-i;
  return lerp(samples[i][key],samples[j][key],t);
}
function angleFromVertical(a,b){
  const v=b.clone().sub(a).normalize();
  return THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(v.dot(new THREE.Vector3(0,1,0)),-1,1)));
}

async function loadOne(url){
  const loader=new BVHLoader();
  const data=await loader.loadAsync(url);
  const root=data.skeleton.bones[0];
  const mixer=new THREE.AnimationMixer(root);
  const action=mixer.clipAction(data.clip);
  action.play();

  const bones={
    hips:findBone(data.skeleton,["hips","hip"]),
    spine:findBone(data.skeleton,["spine","spine1"]),
    chest:findBone(data.skeleton,["spine2","spine3","chest"]),
    leftUp:findBone(data.skeleton,["leftupleg","lhipjoint","leftthigh"]),
    leftLeg:findBone(data.skeleton,["leftleg","leftcalf"]),
    leftFoot:findBone(data.skeleton,["leftfoot"]),
    rightUp:findBone(data.skeleton,["rightupleg","rhipjoint","rightthigh"]),
    rightLeg:findBone(data.skeleton,["rightleg","rightcalf"]),
    rightFoot:findBone(data.skeleton,["rightfoot"])
  };
  const missing=Object.entries(bones).filter(([,b])=>!b).map(([k])=>k);
  if(missing.length)throw new Error(`BVH missing required bones: ${missing.join(",")}`);

  const N=Math.max(180,Math.min(900,Math.ceil(data.clip.duration*60)));
  const all=[];
  for(let i=0;i<N;i++){
    const t=data.clip.duration*i/(N-1);
    mixer.setTime(t);
    root.updateMatrixWorld(true);
    const l=jointFlexionDeg(wp(bones.leftUp),wp(bones.leftLeg),wp(bones.leftFoot));
    const r=jointFlexionDeg(wp(bones.rightUp),wp(bones.rightLeg),wp(bones.rightFoot));
    const knee=(l+r)/2;
    const hip=wp(bones.hips);
    const trunk=angleFromVertical(wp(bones.spine),wp(bones.chest));
    all.push({t,knee,hipY:hip.y,hipZ:hip.z,trunk});
  }

  let peak=0;
  for(let i=1;i<all.length-1;i++){
    if(all[i].knee>all[peak].knee)peak=i;
  }
  if(all[peak].knee<60)throw new Error(`BVH squat prior has insufficient knee flexion: ${all[peak].knee}`);

  let start=peak;
  while(start>1&&all[start].knee>Math.max(18,all[peak].knee*.18))start--;
  const baseY=all[start].hipY,baseZ=all[start].hipZ;
  const samples=[];
  const COUNT=101;
  for(let k=0;k<COUNT;k++){
    const u=k/(COUNT-1);
    const idx=Math.min(peak,Math.round(start+(peak-start)*u));
    const s=all[idx];
    samples.push({
      phase:u,
      kneeFlexDeg:s.knee,
      trunkLeanDeg:s.trunk,
      hipDropRaw:baseY-s.hipY,
      hipForwardRaw:s.hipZ-baseZ
    });
  }

  const maxDrop=Math.max(1e-6,...samples.map(s=>Math.abs(s.hipDropRaw)));
  for(const s of samples){
    s.hipDropNorm=s.hipDropRaw/maxDrop;
    s.hipForwardNorm=s.hipForwardRaw/maxDrop;
  }
  return {source:url,peakFlexDeg:all[peak].knee,samples};
}

export async function loadMotionPriors(){
  const urls=["/motions/cmu/22_14.bvh","/motions/cmu/23_14.bvh"];
  const priors=[];
  for(const url of urls){
    try{priors.push(await loadOne(url));}
    catch(e){console.warn("CMU prior failed",url,e);}
  }
  if(!priors.length)return {squat:null};

  const count=Math.min(...priors.map(p=>p.samples.length));
  const averaged=[];
  for(let i=0;i<count;i++){
    const xs=priors.map(p=>p.samples[i]);
    averaged.push({
      phase:xs[0].phase,
      kneeFlexDeg:xs.reduce((a,s)=>a+s.kneeFlexDeg,0)/xs.length,
      trunkLeanDeg:xs.reduce((a,s)=>a+s.trunkLeanDeg,0)/xs.length,
      hipDropNorm:xs.reduce((a,s)=>a+s.hipDropNorm,0)/xs.length,
      hipForwardNorm:xs.reduce((a,s)=>a+s.hipForwardNorm,0)/xs.length
    });
  }

  return {
    squat:{
      source:"CMU 22_14 + 23_14",
      sample(phase){
        return {
          kneeFlexDeg:interp(averaged,phase,"kneeFlexDeg"),
          trunkLeanDeg:interp(averaged,phase,"trunkLeanDeg"),
          hipDropNorm:interp(averaged,phase,"hipDropNorm"),
          hipForwardNorm:interp(averaged,phase,"hipForwardNorm")
        };
      },
      samples:averaged
    }
  };
}
