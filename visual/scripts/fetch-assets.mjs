import fs from "node:fs/promises";
import path from "node:path";

const manifestPath=new URL("../assets-manifest.json",import.meta.url);
const manifest=JSON.parse(await fs.readFile(manifestPath,"utf8"));
const root=path.resolve("public");

for(const asset of manifest.assets){
  const target=path.join(root,asset.path);
  await fs.mkdir(path.dirname(target),{recursive:true});
  try{
    const st=await fs.stat(target);
    if(st.size>1024){console.log("exists",asset.id,st.size);continue;}
  }catch{}
  const r=await fetch(asset.url,{redirect:"follow"});
  if(!r.ok)throw new Error(`asset download failed ${asset.id}: ${r.status} ${r.statusText}`);
  const bytes=Buffer.from(await r.arrayBuffer());
  if(bytes.length<1024)throw new Error(`asset too small ${asset.id}: ${bytes.length}`);
  await fs.writeFile(target,bytes);
  console.log("downloaded",asset.id,bytes.length);
}
