import fs from "node:fs/promises";
import path from "node:path";

const assets = [
  ["Soldier.glb","https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb"],
  ["Michelle.glb","https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Michelle.glb"]
];

const out = path.resolve("public/models");
await fs.mkdir(out,{recursive:true});
for (const [name,url] of assets) {
  const target=path.join(out,name);
  try { await fs.access(target); console.log("exists",name); continue; } catch {}
  const r=await fetch(url);
  if(!r.ok) throw new Error(`asset download failed ${name}: ${r.status}`);
  await fs.writeFile(target,Buffer.from(await r.arrayBuffer()));
  console.log("downloaded",name);
}
