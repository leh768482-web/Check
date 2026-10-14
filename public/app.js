import { FaceLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22";

const $ = id => document.getElementById(id);
const video = $("camera"), mainCanvas = $("mainCanvas");
const mainCtx = mainCanvas.getContext("2d");
const canvasA = $("canvasA"), canvasB = $("canvasB");
let landmarker = null, cameraStream = null, cameraOn = false, raf = null;
let mainImage = null, sampleA = null, sampleB = null, profileImage = null, lastVideoTime = -1;
let toastTimer;

function toast(msg) {
  $("toast").textContent = msg;
  $("toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("show"), 3000);
}
function status(text, ok=false) {
  $("faceStatus").textContent = text;
  $("faceStatus").style.color = ok ? "#8ff0c1" : "#a5b5d0";
}
async function getLandmarker() {
  if (landmarker) return landmarker;
  status("Đang tải AI…");
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm");
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", delegate: "GPU" },
    runningMode: "VIDEO", numFaces: 1,
    outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false
  });
  return landmarker;
}
function sizeCanvas(canvas, w, h) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(w*dpr));
  canvas.height = Math.max(1, Math.round(h*dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
function drawFace(canvas, source, landmarks) {
  const ctx = canvas.getContext("2d");
  const w = source.videoWidth || source.naturalWidth || source.width;
  const h = source.videoHeight || source.naturalHeight || source.height;
  sizeCanvas(canvas, w, h);
  ctx.clearRect(0,0,w,h);
  ctx.drawImage(source,0,0,w,h);
  if (!landmarks) return;
  const utils = new DrawingUtils(ctx);
  utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {color:"#6ce8f055",lineWidth:.55});
  utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {color:"#6ce8f0",lineWidth:2});
  utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {color:"#ff78c9",lineWidth:1.5});
  utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {color:"#ff78c9",lineWidth:1.5});
  utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, {color:"#ff78c9",lineWidth:1.4});
  utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, {color:"#ff78c9",lineWidth:1.4});
  utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {color:"#ff78c9",lineWidth:1.5});
  utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_NOSE, {color:"#ffb1e2",lineWidth:1.3});
}
function normalizedStats(lm) {
  const dist=(a,b)=>Math.hypot(lm[a].x-lm[b].x,lm[a].y-lm[b].y);
  const xs=lm.map(p=>p.x), ys=lm.map(p=>p.y);
  const width=Math.max(...xs)-Math.min(...xs), height=Math.max(...ys)-Math.min(...ys);
  // Eye landmarks are approximate representative points; values are normalized image ratios.
  const eyeDistance=dist(468 < lm.length ? 468 : 33, 473 < lm.length ? 473 : 263);
  return {eyeDistance, ratio:width/Math.max(height,1e-6), width,height};
}
function setMetrics(lm) {
  if (!lm) {
    $("landmarkCount").innerHTML='— <small>điểm mốc</small>';
    $("eyeDistance").textContent="—"; $("faceRatio").textContent="—";
    $("faceFound").textContent="Không tìm thấy"; status("Không có khuôn mặt");
    return;
  }
  const s=normalizedStats(lm);
  $("landmarkCount").innerHTML=`${lm.length} <small>điểm mốc</small>`;
  $("eyeDistance").textContent=(s.eyeDistance*100).toFixed(1)+"% ảnh";
  $("faceRatio").textContent=s.ratio.toFixed(2);
  $("faceFound").textContent="Đã phát hiện"; status("Đã phát hiện",true);
}
async function detectStill(img) {
  const lmkr=await getLandmarker();
  const oldMode = "IMAGE";
  // Tasks API supports switching modes; still images use a stable timestamp.
  await lmkr.setOptions({runningMode:oldMode});
  const result=lmkr.detect(img);
  await lmkr.setOptions({runningMode:"VIDEO"});
  return result.faceLandmarks?.[0] || null;
}
async function useMainImage(img) {
  stopCamera();
  mainImage=img;
  $("emptyMain").hidden=true;
  try {
    status("Đang phân tích…");
    const lm=await detectStill(img);
    drawFace(mainCanvas,img,lm);
    setMetrics(lm);
    if (!lm) toast("Không tìm thấy khuôn mặt. Hãy thử ảnh rõ mặt, đủ sáng.");
  } catch(e) { console.error(e); toast("Không tải được mô hình AI. Kiểm tra kết nối Internet rồi thử lại."); status("Lỗi phân tích"); }
}
function loadFile(file, callback) {
  if (!file) return;
  if (!file.type.startsWith("image/")) { toast("Vui lòng chọn tệp ảnh hợp lệ."); return; }
  if (file.size > 12*1024*1024) { toast("Ảnh tối đa 12 MB."); return; }
  const url=URL.createObjectURL(file), img=new Image();
  img.onload=()=>{URL.revokeObjectURL(url); callback(img);};
  img.onerror=()=>{URL.revokeObjectURL(url);toast("Không đọc được ảnh này.");};
  img.src=url;
}
function drawSample(canvas,img,lm) {
  const ctx=canvas.getContext("2d"), w=img.naturalWidth,h=img.naturalHeight;
  sizeCanvas(canvas,w,h); ctx.drawImage(img,0,0,w,h);
  if(lm){
    const utils=new DrawingUtils(ctx);
    utils.drawConnectors(lm,FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,{color:"#6ce8f0",lineWidth:2});
    utils.drawConnectors(lm,FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,{color:"#ff78c9",lineWidth:1.5});
    utils.drawConnectors(lm,FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,{color:"#ff78c9",lineWidth:1.5});
    utils.drawConnectors(lm,FaceLandmarker.FACE_LANDMARKS_LIPS,{color:"#ff78c9",lineWidth:1.5});
    for(let i=0;i<lm.length;i+=3){ctx.fillStyle="#6ce8f0";ctx.beginPath();ctx.arc(lm[i].x*w,lm[i].y*h,1.2,0,Math.PI*2);ctx.fill();}
  }
}
async function prepareSample(file,which) {
  loadFile(file,async img=>{
    try {
      await getLandmarker();
      const lm=await detectStill(img);
      const item={img,lm};
      if(which==="A"){sampleA=item;drawSample(canvasA,img,lm);$("emptyA").hidden=true;$("aState").textContent=lm?"Đã phát hiện":"Không có mặt";}
      else {sampleB=item;drawSample(canvasB,img,lm);$("emptyB").hidden=true;$("bState").textContent=lm?"Đã phát hiện":"Không có mặt";}
      if(!lm)toast("Ảnh đã tải nhưng không phát hiện được khuôn mặt.");
    }catch(e){console.error(e);toast("Lỗi tải mô hình AI. Kiểm tra mạng rồi thử lại.");}
  });
}
// Compares normalized landmark configurations. This is NOT face identity recognition.
function similarity(lmA,lmB) {
  if(!lmA||!lmB||lmA.length!==lmB.length) return null;
  const center=lm=>{const xs=lm.map(p=>p.x),ys=lm.map(p=>p.y);return {x:(Math.min(...xs)+Math.max(...xs))/2,y:(Math.min(...ys)+Math.max(...ys))/2,w:Math.max(...xs)-Math.min(...xs),h:Math.max(...ys)-Math.min(...ys)}};
  const a=center(lmA),b=center(lmB);
  const norm=(lm,c)=>lm.map(p=>({x:(p.x-c.x)/Math.max(c.w,1e-5),y:(p.y-c.y)/Math.max(c.h,1e-5)}));
  const na=norm(lmA,a),nb=norm(lmB,b);
  let sum=0;
  for(let i=0;i<na.length;i++) sum+=Math.hypot(na[i].x-nb[i].x,na[i].y-nb[i].y);
  const mean=sum/na.length;
  // A bounded visual-shape similarity heuristic, not a calibrated or validated biometric score.
  return Math.max(0,Math.min(100,Math.round(100*Math.exp(-mean*1.8))));
}
function renderScore(container,score,sourceLabel="hai ảnh") {
  if(score===null){container.innerHTML='<div class="result-placeholder">Không đủ dữ liệu để so sánh. Hãy chọn hai ảnh có khuôn mặt rõ ràng.</div>';return;}
  const high=score>=80, threshold=$("threshold").checked;
  if(threshold&&!high){container.innerHTML='<div class="result-placeholder">Điểm dưới ngưỡng 80% nên đã ẩn kết quả theo tùy chọn. Bạn có thể bỏ chọn “Chỉ hiện trạng thái…” để xem.</div>';return;}
  const state=high?"Tương đồng hình học cao":score>=55?"Cần xem xét":"Tương đồng hình học thấp";
  container.innerHTML=`<div class="score-card"><div class="score">${score}<small>%</small></div><div><h3>${state}</h3><p>So sánh ${sourceLabel}. Đây là điểm hình học thử nghiệm, không phải xác suất cùng danh tính. Tư thế, biểu cảm, ánh sáng và chất lượng ảnh có thể ảnh hưởng kết quả.</p><div class="bar"><span style="width:${score}%"></span></div></div></div>`;
}
$("startCamera").addEventListener("click",async()=>{
  try{
    if(!navigator.mediaDevices?.getUserMedia) throw new Error("Trình duyệt không hỗ trợ camera hoặc trang chưa dùng HTTPS.");
    await getLandmarker();
    cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:1280},height:{ideal:720}},audio:false});
    video.srcObject=cameraStream; video.hidden=false; await video.play();
    cameraOn=true; $("capture").disabled=false; $("stopCamera").disabled=false;
    $("emptyMain").hidden=true; $("cameraStatus").textContent="Camera đang hoạt động. Bấm Chụp ảnh để phân tích.";
    $("startCamera").disabled=true; lastVideoTime=-1; cameraLoop();
  }catch(e){console.error(e);toast(e.message.includes("Permission")?"Bạn chưa cấp quyền camera.":e.message||"Không thể mở camera.");$("cameraStatus").textContent="Không mở được camera: kiểm tra quyền trình duyệt và HTTPS.";}
});
function cameraLoop(){
  if(!cameraOn||video.readyState<2)return;
  if(video.currentTime!==lastVideoTime){
    lastVideoTime=video.currentTime;
    try{
      const result=landmarker.detectForVideo(video,performance.now());
      const lm=result.faceLandmarks?.[0]||null;
      drawFace(mainCanvas,video,lm); setMetrics(lm);
    }catch(e){console.warn(e);}
  }
  raf=requestAnimationFrame(cameraLoop);
}
function stopCamera(){
  cameraOn=false;if(raf)cancelAnimationFrame(raf);
  if(cameraStream){cameraStream.getTracks().forEach(t=>t.stop());cameraStream=null;}
  video.srcObject=null;video.hidden=true;
  $("capture").disabled=true;$("stopCamera").disabled=true;$("startCamera").disabled=false;
}
$("stopCamera").addEventListener("click",stopCamera);
$("capture").addEventListener("click",async()=>{
  if(!cameraOn)return;
  const canvas=document.createElement("canvas");canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  canvas.getContext("2d").drawImage(video,0,0);
  const img=new Image();img.onload=()=>useMainImage(img);img.src=canvas.toDataURL("image/jpeg",.92);
});
$("mainUpload").addEventListener("change",e=>loadFile(e.target.files[0],useMainImage));
$("uploadA").addEventListener("change",e=>prepareSample(e.target.files[0],"A"));
$("uploadB").addEventListener("change",e=>prepareSample(e.target.files[0],"B"));
$("compareBtn").addEventListener("click",()=>{
  if(!sampleA||!sampleB){toast("Hãy chọn cả ảnh A và ảnh B.");return;}
  if(!sampleA.lm||!sampleB.lm){renderScore($("compareResult"),null);return;}
  renderScore($("compareResult"),similarity(sampleA.lm,sampleB.lm),"ảnh A và ảnh B");
});
$("profileImage").addEventListener("change",e=>{
  loadFile(e.target.files[0],async img=>{
    try{await getLandmarker();const lm=await detectStill(img);profileImage={img,lm};if(!lm)toast("Không phát hiện khuôn mặt trong ảnh đại diện.");}
    catch(e){toast("Không thể phân tích ảnh đại diện.");}
  });
});
$("profileCompare").addEventListener("click",()=>{
  const url=$("profileUrl").value.trim(), platform=$("platform").value;
  if(!sampleA?.lm){toast("Hãy chọn ảnh A trước.");return;}
  if(!profileImage?.lm){toast("Hãy tải ảnh đại diện có khuôn mặt.");return;}
  if(!url){toast("Hãy nhập URL hồ sơ do bạn cung cấp.");return;}
  try{const parsed=new URL(url);if(!["http:","https:"].includes(parsed.protocol))throw new Error();}
  catch{toast("URL hồ sơ không hợp lệ.");return;}
  const score=similarity(sampleA.lm,profileImage.lm);
  renderScore($("profileResult"),score,`ảnh A và ảnh đại diện ${platform}`);
  const link=document.createElement("a");link.href=url;link.target="_blank";link.rel="noopener noreferrer";link.textContent="Mở hồ sơ nguồn ↗";link.className="btn";link.style.marginTop="10px";$("profileResult").appendChild(link);
});
window.addEventListener("beforeunload",stopCamera);

