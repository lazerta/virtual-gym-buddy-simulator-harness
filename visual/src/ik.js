import * as THREE from "three";

const wp=o=>o.getWorldPosition(new THREE.Vector3());
const wq=o=>o.getWorldQuaternion(new THREE.Quaternion());

function aimBoneAt(bone,child,target){
  bone.updateWorldMatrix(true,true);
  const a=wp(bone), b=wp(child), current=b.sub(a).normalize(), desired=target.clone().sub(a).normalize();
  if(current.lengthSq()<1e-8||desired.lengthSq()<1e-8) return;
  const delta=new THREE.Quaternion().setFromUnitVectors(current,desired);
  const desiredWorld=delta.multiply(wq(bone));
  const parentWorld=bone.parent?wq(bone.parent):new THREE.Quaternion();
  bone.quaternion.copy(parentWorld.invert().multiply(desiredWorld));
  bone.updateWorldMatrix(true,true);
}

export function solveTwoBoneIK({upper,lower,end,target,pole}){
  if(!upper||!lower||!end) return {ok:false,error:Infinity};
  upper.updateWorldMatrix(true,true);
  const A=wp(upper), B=wp(lower), C=wp(end);
  const l1=A.distanceTo(B), l2=B.distanceTo(C);
  if(l1<1e-5||l2<1e-5) return {ok:false,error:Infinity};
  const toT=target.clone().sub(A); let d=toT.length();
  const dir=toT.clone().normalize();
  d=Math.max(Math.abs(l1-l2)+1e-4,Math.min(l1+l2-1e-4,d));
  const poleVec=(pole||A.clone().add(new THREE.Vector3(0,0,1))).clone().sub(A);
  let perp=poleVec.sub(dir.clone().multiplyScalar(poleVec.dot(dir)));
  if(perp.lengthSq()<1e-8) perp=new THREE.Vector3(0,0,1).cross(dir);
  if(perp.lengthSq()<1e-8) perp=new THREE.Vector3(1,0,0).cross(dir);
  perp.normalize();
  const x=(l1*l1+d*d-l2*l2)/(2*d);
  const h=Math.sqrt(Math.max(0,l1*l1-x*x));
  const elbow=A.clone().add(dir.clone().multiplyScalar(x)).add(perp.multiplyScalar(h));
  aimBoneAt(upper,lower,elbow);
  aimBoneAt(lower,end,target);
  const err=wp(end).distanceTo(target);
  return {ok:true,error:err,mid:elbow};
}

export function jointFlexionDeg(a,b,c){
  if(!a||!b||!c) return null;
  const ba=a.clone().sub(b).normalize(), bc=c.clone().sub(b).normalize();
  return 180-THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(ba.dot(bc),-1,1)));
}

export function distanceForFlexion(l1,l2,flexDeg){
  const included=THREE.MathUtils.degToRad(180-flexDeg);
  return Math.sqrt(Math.max(1e-8,l1*l1+l2*l2-2*l1*l2*Math.cos(included)));
}
