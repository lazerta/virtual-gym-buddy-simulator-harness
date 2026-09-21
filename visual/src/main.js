import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {AVATARS,EXERCISES,FORMS} from "./catalog.js";
import {resolveRig,captureRest,captureRootRest,normalizeAvatar,measureRig} from "./rig.js";
import {createCommercialGym} from "./equipment.js";
import {applyMotion,cyclePhase} from "./motion.js";
import {collectGroundTruth} from "./groundTruth.js";

const app=document.getElementById("app"),hud=document.getElementById("hud");
const avatarSel=document.getElementById("avatar"),exerciseSel=document.getElementById("exercise"),formSel=document.getElementById("form");
Object.values(AVATARS).forEach(a=>avatarSel.add(new Option(a.label,a.id)));Object.values(EXERCISES).forEach(e=>exerciseSel.add(new Option(e.label,e.id)));Object.values(FORMS).forEach(f=>formSel.add(new Option(f.label,f.id)));
const scene=new THREE.Scene();scene.background=new THREE.Color(0x202226);scene.fog=new THREE.Fog(0x202226,10,24);
const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,.1,100);camera.position.set(4.2,2.6,5.4);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance"});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;app.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,1,0);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x30343a,1.7));const sun=new THREE.DirectionalLight(0xffffff,3.2);sun.position.set(4,7,3);sun.castShadow=true;scene.add(sun);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(18,18),new THREE.MeshStandardMaterial({color:0x3b3e41,roughness:.95}));floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;scene.add(floor);const grid=new THREE.GridHelper(18,18,0x4a4d50,0x333638);grid.position.y=.002;scene.add(grid);
const gym=createCommercialGym(scene),loader=new GLTFLoader();let model=null,rig={},rigRest={},rootRest=null,metrics=null,generation=0,currentEquipment=null;
function configureEquipment(){Object.values(gym).forEach(s=>s.group.visible=false);const key=EXERCISES[exerciseSel.value].equipment;currentEquipment=gym[key];if(currentEquipment)currentEquipment.group.visible=true}
async function loadAvatar(){const gen=++generation,spec=AVATARS[avatarSel.value],gltf=await loader.loadAsync(spec.url);if(gen!==generation)return;if(model)scene.remove(model);model=gltf.scene;normalizeAvatar(model,spec.targetHeight);model.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true}});scene.add(model);rig=resolveRig(model);const required=["hips","leftUpperArm","leftLowerArm","leftHand","rightUpperArm","rightLowerArm","rightHand","leftUpperLeg","leftLowerLeg","leftFoot","rightUpperLeg","rightLowerLeg","rightFoot"];const missing=required.filter(k=>!rig[k]);if(missing.length)throw new Error(`avatar rig missing: ${missing.join(", ")}`);rigRest=captureRest(rig);rootRest=captureRootRest(model);metrics=measureRig(model,rig)}
avatarSel.onchange=loadAvatar;exerciseSel.onchange=configureEquipment;avatarSel.value="soldier";exerciseSel.value="smith_squat";formSel.value="correct";configureEquipment();await loadAvatar();
const clock=new THREE.Clock();renderer.setAnimationLoop(()=>{if(!model||!metrics)return;const phase=cyclePhase(clock.getElapsedTime());const truth=applyMotion({exercise:exerciseSel.value,form:formSel.value,phase,model,rig,rigRest,rootRest,metrics,station:currentEquipment});const gt=collectGroundTruth(model,rig,{avatar_id:avatarSel.value,exercise_id:exerciseSel.value,form_id:formSel.value,phase:truth.phase,issues:truth.issues,constraint_error_m:truth.constraint_error_m,measured:truth.measured});window.__GYM_BUDDY_GT__=gt;hud.textContent=`Gym Buddy Scientific Visual Harness
avatar: ${gt.avatar_id} (${metrics.height.toFixed(2)} m)
exercise: ${gt.exercise_id}
form: ${gt.form_id}
phase: ${gt.phase.toFixed(3)}
constraint error: ${(gt.constraint_error_m*100).toFixed(1)} cm
L elbow: ${gt.joint_angles.left_elbow_flex_deg?.toFixed(1)??"n/a"}°  L knee: ${gt.joint_angles.left_knee_flex_deg?.toFixed(1)??"n/a"}°
issues: ${JSON.stringify(gt.issues)}`;controls.update();renderer.render(scene,camera)});
addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
