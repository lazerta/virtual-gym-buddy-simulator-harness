import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {AVATARS,EXERCISES,FORMS} from "./catalog.js";
import {resolveRig,captureRest} from "./rig.js";
import {createCommercialGym} from "./equipment.js";
import {applyMotion,cyclePhase} from "./motion.js";
import {collectGroundTruth} from "./groundTruth.js";

const app=document.getElementById("app"),hud=document.getElementById("hud");
const avatarSel=document.getElementById("avatar"),exerciseSel=document.getElementById("exercise"),formSel=document.getElementById("form");
Object.values(AVATARS).forEach(a=>avatarSel.add(new Option(a.label,a.id)));
Object.values(EXERCISES).forEach(e=>exerciseSel.add(new Option(e.label,e.id)));
Object.values(FORMS).forEach(f=>formSel.add(new Option(f.label,f.id)));

const scene=new THREE.Scene();scene.background=new THREE.Color(0x202226);scene.fog=new THREE.Fog(0x202226,10,24);
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,.1,100);camera.position.set(4.2,2.6,5.4);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance"});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;app.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,1,0);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x30343a,1.7));const sun=new THREE.DirectionalLight(0xffffff,3.2);sun.position.set(4,7,3);sun.castShadow=true;scene.add(sun);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(18,18),new THREE.MeshStandardMaterial({color:0x3b3e41,roughness:.95}));floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;scene.add(floor);
for(let x=-8;x<=8;x+=1){const l=new THREE.GridHelper(18,18,0x4a4d50,0x333638);l.position.y=.002;scene.add(l);break}
const gym=createCommercialGym(scene);
const loader=new GLTFLoader();let model=null,rig={},rest={},generation=0,currentEquipment=null;

function configureEquipment(){
  Object.values(gym).forEach(s=>s.group.visible=false);
  const key=EXERCISES[exerciseSel.value].equipment;currentEquipment=gym[key];if(currentEquipment)currentEquipment.group.visible=true;
}
async function loadAvatar(){
  const gen=++generation,spec=AVATARS[avatarSel.value],gltf=await loader.loadAsync(spec.url);if(gen!==generation)return;
  if(model)scene.remove(model);model=gltf.scene;model.position.set(0,0,.15);model.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true}});scene.add(model);rig=resolveRig(model);rest=captureRest(rig);
}
avatarSel.onchange=loadAvatar;exerciseSel.onchange=configureEquipment;
avatarSel.value="soldier";exerciseSel.value="smith_squat";formSel.value="correct";configureEquipment();await loadAvatar();

const clock=new THREE.Clock();
renderer.setAnimationLoop(()=>{
  if(!model)return;model.rotation.set(0,0,0);
  const phase=cyclePhase(clock.getElapsedTime());
  const truth=applyMotion({exercise:exerciseSel.value,form:formSel.value,phase,model,rig,rest,equipment:currentEquipment});
  const gt=collectGroundTruth(model,rig,{avatar_id:avatarSel.value,exercise_id:exerciseSel.value,form_id:formSel.value,phase:truth.phase,issues:truth.issues});
  window.__GYM_BUDDY_GT__=gt;
  hud.textContent=`Gym Buddy Visual Harness
avatar: ${gt.avatar_id}
exercise: ${gt.exercise_id}
form: ${gt.form_id}
phase: ${gt.phase.toFixed(3)}
joints: ${Object.keys(gt.joints3d).length}
issues: ${JSON.stringify(gt.issues)}
GPU: ${renderer.getContext().getParameter(renderer.getContext().RENDERER)}`;
  controls.update();renderer.render(scene,camera);
});
addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
