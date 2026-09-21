import {defineConfig} from "@playwright/test";

export default defineConfig({
  testDir:"./tests",
  timeout:120000,
  workers:1,
  use:{
    baseURL:"http://127.0.0.1:5173",
    headless:true,
    viewport:{width:1280,height:800},
    launchOptions:{args:["--use-angle=swiftshader","--enable-webgl"]}
  },
  webServer:{
    command:"npm run dev -- --port 5173",
    url:"http://127.0.0.1:5173",
    reuseExistingServer:false,
    timeout:120000
  }
});
