import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {AVATARS,EXERCISES,FORMS} from "./catalog.js";
import {resolveRig,captureRest} from "./rig.js";
import {createSmithMachine,createDumbbells} from "./equipment.js";
import {applyMotion,cyclePhase} from "./motion.js";
import {collectGroundTruth} from "./groundTruth.js";

const app=document.getElementById("app"), hud=document.getElementById("hud");
const avatarSel=document.getElementById("avatar"), exerciseSel=document.getElementById("exercise"), formSel=document.getElementById("form");
for(const a of Object.values(AVATARS)) avatarSel.add(new Option(a.label,a.id));
for(const e of Object.values(EXERCISES)) exerciseSel.add(new Option(e.label,e.id));
for(const f of Object.values(FORMS)) formSel.add(new Option(f.label,f.id));

const scene=new THREE.Scene(); scene.background=new THREE.Color(0x202226);
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,.1,100); camera.position.set(4,2.5,5);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance"});
renderer.setSize(innerWidth,innerHeight); renderer.setPixelRatio(devicePixelRatio); renderer.shadowMap.enabled=true; app.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement); controls.target.set(0,1,0); controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x444444,2));
const sun=new THREE.DirectionalLight(0xffffff,3); sun.position.set(4,7,3); sun.castShadow=true; scene.add(sun);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(14,14),new THREE.MeshStandardMaterial({color:0x454545,roughness:.9})); floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor);

const smith=createSmithMachine(scene); const db=createDumbbells(scene); db.group.visible=false;
const loader=new GLTFLoader();
let model=null,rig={},rest={},generation=0;

function configureEquipment(){
  const ex=EXERCISES[exerciseSel.value];
  smith.group.visible=ex.equipment==="smith"; db.group.visible=ex.equipment==="dumbbells";
}
async function loadAvatar(){
  const gen=++generation, spec=AVATARS[avatarSel.value];
  const gltf=await loader.loadAsync(spec.url); if(gen!==generation)return;
  if(model) scene.remove(model);
  model=gltf.scene; model.position.set(0,0,.15); model.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true}});
  scene.add(model); rig=resolveRig(model); rest=captureRest(rig);
}
avatarSel.onchange=loadAvatar; exerciseSel.onchange=configureEquipment; formSel.onchange=()=>{};
avatarSel.value="soldier"; exerciseSel.value="smith_squat"; formSel.value="correct";
configureEquipment(); await loadAvatar();

const clock=new THREE.Clock();
renderer.setAnimationLoop(()=>{
  if(!model)return;
  const phase=cyclePhase(clock.getElapsedTime());
  const eq=smith.group.visible?smith:db;
  const truth=applyMotion({exercise:exerciseSel.value,form:formSel.value,phase,model,rig,rest,equipment:eq});
  const gt=collectGroundTruth(model,rig,{avatar_id:avatarSel.value,exercise_id:exerciseSel.value,form_id:formSel.value,phase:truth.phase,issues:truth.issues});
  window.__GYM_BUDDY_GT__=gt;
  hud.textContent=`Gym Buddy Visual Harness
avatar: ${gt.avatar_id}
exercise: ${gt.exercise_id}
form: ${gt.form_id}
phase: ${gt.phase.toFixed(3)}
joints: ${Object.keys(gt.joints3d).length}
issues: ${JSON.stringify(gt.issues)}`;
  controls.update(); renderer.render(scene,camera);
});
addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
