import * as THREE from "three";
import {EXERCISES} from "../src/catalog.js";
import {BIOMECH} from "../src/biomechanics.js";
import {solveTwoBoneIK} from "../src/ik.js";

const ids=Object.keys(EXERCISES);
if(ids.length!==10) throw new Error(`expected 10 exercises, got ${ids.length}`);
for(const id of ids) if(!BIOMECH[id]) throw new Error(`missing biomechanics spec for ${id}`);

const root=new THREE.Group();
const upper=new THREE.Bone(), lower=new THREE.Bone(), end=new THREE.Bone();
root.add(upper);upper.add(lower);lower.add(end);
lower.position.set(0,-1,0);end.position.set(0,-1,0);
root.updateMatrixWorld(true);
const target=new THREE.Vector3(.8,-1.5,.2), pole=new THREE.Vector3(0,0,1);
const solved=solveTwoBoneIK({upper,lower,end,target,pole});
root.updateMatrixWorld(true);
const actual=end.getWorldPosition(new THREE.Vector3());
const error=actual.distanceTo(target);
if(!solved.ok||error>.01) throw new Error(`IK error too high: ${error}`);

console.log(JSON.stringify({ok:true,exercise_count:ids.length,ik_error_m:error}));
