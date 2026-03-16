// ============================================================
// WebGL2 PBR Material Viewer
// Cook-Torrance BRDF · GGX NDF · Smith-Schlick Geometry
// Schlick Fresnel · 3-point lighting · ACES tone map
// ============================================================

const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
if (!gl) { alert('WebGL2 not supported'); }

// ─── STATE ───────────────────────────────────────────────
let state = {
  mesh: 'sphere',
  albedo: [1.0, 0.843, 0.0],
  metallic: 1.0,
  roughness: 0.2,
  ao: 1.0,
  emissive: 0.0,
  lights: [
    { pos: [3, 4, 3],   color: [1.0, 1.0, 1.0], intensity: 3.0 },
    { pos: [-4, 2, -2], color: [0.53, 0.67, 1.0], intensity: 1.5 },
    { pos: [0, -3, 4],  color: [1.0, 0.53, 0.27], intensity: 0.8 },
  ],
  tonemapping: true,
  gamma: true,
  exposure: 1.0,
  channel: 0,
  autoRotate: true,
  rotSpeed: 0.4,
  ambientColor: [0.08, 0.09, 0.12],
  // Parallax
  parallaxEnabled: true,
  parallaxScale: 0.06,
  parallaxSteps: 32,
  parallaxShadow: true,
  heightPattern: 'bricks',
};

let camera = { theta: 0.4, phi: 0.8, radius: 2.8, panX: 0, panY: 0 };
let mouse = { down: false, rightDown: false, x: 0, y: 0 };
let meshBuffers = {};
let program;
let heightTex;
let frameCount = 0, lastFpsTime = performance.now(), fps = 0;
let autoRotAngle = 0;

// ─── SHADERS ─────────────────────────────────────────────
const VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
layout(location=3) in vec3 aTangent;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;
uniform vec3 uCamPos;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out mat3 vTBN;
out vec3 vTangentViewDir; // view dir in tangent space for parallax

void main() {
  vec4 worldPos = uModel * vec4(aPos, 1.0);
  vWorldPos = worldPos.xyz;
  vNormal = normalize(uNormalMat * aNormal);
  vUV = aUV;

  vec3 T = normalize(uNormalMat * aTangent);
  vec3 N = vNormal;
  T = normalize(T - dot(T, N) * N);
  vec3 B = cross(N, T);
  vTBN = mat3(T, B, N);

  // Transform view direction into tangent space
  vec3 viewDir = uCamPos - worldPos.xyz;
  vTangentViewDir = vec3(dot(viewDir, T), dot(viewDir, B), dot(viewDir, N));

  gl_Position = uProj * uView * worldPos;
}`;

const FS = `#version 300 es
precision highp float;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in mat3 vTBN;
in vec3 vTangentViewDir;

out vec4 fragColor;

// Material
uniform vec3  uAlbedo;
uniform float uMetallic;
uniform float uRoughness;
uniform float uAO;
uniform float uEmissive;

// Lights
uniform vec3  uLightPos[3];
uniform vec3  uLightColor[3];
uniform float uLightIntensity[3];
uniform vec3  uAmbient;

// Camera
uniform vec3  uCamPos;

// Post
uniform bool  uTonemap;
uniform bool  uGamma;
uniform float uExposure;
uniform int   uChannel;

// Parallax
uniform sampler2D uHeightMap;
uniform float     uParallaxScale;
uniform int       uParallaxSteps;
uniform bool      uParallaxEnabled;
uniform bool      uParallaxShadow;

const float PI = 3.14159265359;

// ── Height map sampling ──────────────────────────────────
float sampleHeight(vec2 uv) {
  return texture(uHeightMap, uv).r;
}

// ── Steep Parallax Occlusion Mapping ────────────────────
vec2 parallaxOcclusionMapping(vec2 uv, vec3 viewDirTS) {
  if (!uParallaxEnabled || uParallaxScale < 0.0001) return uv;

  // Number of layers scales with view angle — more layers at grazing
  float NdotV = abs(viewDirTS.z);
  float numLayers = mix(float(uParallaxSteps) * 2.0, float(uParallaxSteps), NdotV);
  numLayers = clamp(numLayers, 4.0, 128.0);

  float layerDepth = 1.0 / numLayers;
  float currentLayerDepth = 0.0;

  // UV shift per layer — xy in tangent space, scaled by height
  vec2 P = (viewDirTS.xy / viewDirTS.z) * uParallaxScale;
  vec2 deltaUV = P / numLayers;

  vec2  currentUV        = uv;
  float currentHeight    = sampleHeight(currentUV);

  // March through layers until surface is hit
  for (int i = 0; i < 256; i++) {
    if (float(i) >= numLayers) break;
    if (currentLayerDepth >= currentHeight) break;
    currentUV          -= deltaUV;
    currentHeight       = sampleHeight(currentUV);
    currentLayerDepth  += layerDepth;
  }

  // Binary search refinement (4 steps) for sub-layer precision
  vec2 prevUV     = currentUV + deltaUV;
  float afterH    = currentHeight - currentLayerDepth;
  float beforeH   = sampleHeight(prevUV) - (currentLayerDepth - layerDepth);
  float weight    = afterH / (afterH - beforeH);
  return mix(currentUV, prevUV, weight);
}

// ── Parallax self-shadow ─────────────────────────────────
float parallaxSoftShadow(vec2 uv, vec3 lightDirTS, float initialHeight) {
  if (!uParallaxShadow) return 1.0;
  if (lightDirTS.z <= 0.0) return 0.4; // light below surface

  float shadowFactor = 1.0;
  int   shadowSteps  = uParallaxSteps / 2;
  float stepSize     = 1.0 / float(shadowSteps);
  vec2  stepUV       = (lightDirTS.xy / lightDirTS.z) * uParallaxScale * stepSize;
  float stepDepth    = stepSize;

  vec2  sUV          = uv + stepUV;
  float layerH       = initialHeight + stepDepth;

  for (int i = 0; i < 64; i++) {
    if (i >= shadowSteps) break;
    float h = sampleHeight(sUV);
    if (h > layerH) {
      // Soft penumbra: earlier hits cast softer shadows
      float softness = 1.0 - clamp(float(i) / float(shadowSteps), 0.0, 1.0);
      shadowFactor = min(shadowFactor, mix(0.3, 1.0, 1.0 - softness * 0.6));
    }
    sUV    += stepUV;
    layerH += stepDepth;
  }
  return shadowFactor;
}

// ── Normal from height map (central difference) ──────────
vec3 heightToNormal(vec2 uv) {
  float eps = 0.002;
  float hL = sampleHeight(uv - vec2(eps, 0.0));
  float hR = sampleHeight(uv + vec2(eps, 0.0));
  float hD = sampleHeight(uv - vec2(0.0, eps));
  float hU = sampleHeight(uv + vec2(0.0, eps));
  vec3 n = normalize(vec3((hL - hR) / (2.0 * eps) * uParallaxScale * 4.0,
                          (hD - hU) / (2.0 * eps) * uParallaxScale * 4.0,
                          1.0));
  return n;
}

// ── BRDF functions ───────────────────────────────────────
float D_GGX(float NdotH, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

float G_SchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0;
  float k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
  return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

vec3 F_Schlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 ACES(vec3 x) {
  float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
  vec3 V = normalize(uCamPos - vWorldPos);
  vec3 viewDirTS = normalize(vTangentViewDir);

  // ── Parallax UV offset ──
  vec2 uv = parallaxOcclusionMapping(vUV, viewDirTS);
  float surfaceHeight = sampleHeight(uv);

  // ── Derive perturbed normal from height map ──
  vec3 N_geo = normalize(vNormal);
  vec3 N;
  if (uParallaxEnabled && uParallaxScale > 0.0001) {
    vec3 nTS = heightToNormal(uv);         // normal in tangent space
    N = normalize(vTBN * nTS);            // world space
  } else {
    N = N_geo;
  }

  vec3 albedo     = uAlbedo;
  float metallic  = uMetallic;
  float roughness = max(uRoughness, 0.02);
  float ao        = uAO;

  vec3 F0 = mix(vec3(0.04), albedo, metallic);

  // ── Debug channels ──
  if (uChannel == 1) { fragColor = vec4(albedo, 1.0); return; }
  if (uChannel == 2) { fragColor = vec4(N * 0.5 + 0.5, 1.0); return; }
  if (uChannel == 3) { fragColor = vec4(vec3(metallic), 1.0); return; }
  if (uChannel == 4) { fragColor = vec4(vec3(roughness), 1.0); return; }
  if (uChannel == 5) { fragColor = vec4(vec3(ao), 1.0); return; }
  if (uChannel == 6) { fragColor = vec4(vec3(surfaceHeight), 1.0); return; }

  // ── Cook-Torrance PBR with parallax shadow ──
  vec3 Lo = vec3(0.0);
  for (int i = 0; i < 3; i++) {
    vec3 L_world = normalize(uLightPos[i] - vWorldPos);
    vec3 H = normalize(V + L_world);
    float dist = length(uLightPos[i] - vWorldPos);
    float attn = 1.0 / (dist * dist);
    vec3 radiance = uLightColor[i] * uLightIntensity[i] * attn;

    float NdotV = max(dot(N, V), 0.0001);
    float NdotL = max(dot(N, L_world), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // Parallax self-shadow: compute light dir in tangent space
    float shadow = 1.0;
    if (uParallaxEnabled) {
      vec3 L_TS = normalize(vec3(dot(L_world, vTBN[0]),
                                 dot(L_world, vTBN[1]),
                                 dot(L_world, vTBN[2])));
      shadow = parallaxSoftShadow(uv, L_TS, surfaceHeight);
    }

    float NDF = D_GGX(NdotH, roughness);
    float G   = G_Smith(NdotV, NdotL, roughness);
    vec3  F   = F_Schlick(HdotV, F0);

    vec3  num   = NDF * G * F;
    float denom = 4.0 * NdotV * NdotL + 0.0001;
    vec3  spec  = num / denom;

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    Lo += (kD * albedo / PI + spec) * radiance * NdotL * shadow;
  }

  // Ambient IBL approximation
  vec3 kS_amb = F_Schlick(max(dot(N, V), 0.0), F0);
  vec3 kD_amb = (vec3(1.0) - kS_amb) * (1.0 - metallic);
  vec3 ambient = (kD_amb * albedo + kS_amb * 0.15) * uAmbient * ao;

  vec3 emissive = albedo * uEmissive;
  vec3 color = ambient + Lo + emissive;

  color *= uExposure;
  if (uTonemap) color = ACES(color);
  if (uGamma)   color = pow(color, vec3(1.0/2.2));

  fragColor = vec4(color, 1.0);
}`;

// ─── SHADER COMPILE ──────────────────────────────────────
function compileShader(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    console.error(gl.getShaderInfoLog(s));
  return s;
}

function createProgram(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compileShader(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compileShader(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    console.error(gl.getProgramInfoLog(p));
  return p;
}

// ─── MATH UTILS ──────────────────────────────────────────
function mat4() { return new Float32Array(16); }

function mat4Identity(m) {
  m.fill(0); m[0]=m[5]=m[10]=m[15]=1; return m;
}

function mat4Mul(out, a, b) {
  for (let i=0;i<4;i++) for (let j=0;j<4;j++) {
    let s=0; for(let k=0;k<4;k++) s+=a[i*4+k]*b[k*4+j];
    out[i*4+j]=s;
  }
  return out;
}

function mat4Perspective(m, fov, aspect, near, far) {
  mat4Identity(m);
  const f = 1.0/Math.tan(fov/2);
  m[0]=f/aspect; m[5]=f;
  m[10]=(far+near)/(near-far); m[11]=-1;
  m[14]=(2*far*near)/(near-far); m[15]=0;
  return m;
}

function mat4LookAt(m, eye, center, up) {
  const f = normalize3(sub3(center, eye));
  const r = normalize3(cross3(f, up));
  const u = cross3(r, f);
  mat4Identity(m);
  m[0]=r[0]; m[4]=r[1]; m[8]=r[2];
  m[1]=u[0]; m[5]=u[1]; m[9]=u[2];
  m[2]=-f[0]; m[6]=-f[1]; m[10]=-f[2];
  m[12]=-dot3(r,eye); m[13]=-dot3(u,eye); m[14]=dot3(f,eye);
  return m;
}

function mat4Rotate(m, angle, axis) {
  const [x,y,z] = normalize3(axis);
  const c=Math.cos(angle), s=Math.sin(angle), t=1-c;
  mat4Identity(m);
  m[0]=t*x*x+c; m[4]=t*x*y-s*z; m[8]=t*x*z+s*y;
  m[1]=t*x*y+s*z; m[5]=t*y*y+c; m[9]=t*y*z-s*x;
  m[2]=t*x*z-s*y; m[6]=t*y*z+s*x; m[10]=t*z*z+c;
  return m;
}

function mat3NormalFromMat4(m4) {
  // upper-left 3x3 inverse-transpose
  const a=m4[0],b=m4[1],c=m4[2],d=m4[4],e=m4[5],f=m4[6],g=m4[8],h=m4[9],i=m4[10];
  const det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
  const inv=1/det;
  return new Float32Array([
    (e*i-f*h)*inv, (f*g-d*i)*inv, (d*h-e*g)*inv,
    (c*h-b*i)*inv, (a*i-c*g)*inv, (b*g-a*h)*inv,
    (b*f-c*e)*inv, (c*d-a*f)*inv, (a*e-b*d)*inv
  ]);
}

function normalize3(v) { const l=Math.sqrt(v[0]**2+v[1]**2+v[2]**2); return [v[0]/l,v[1]/l,v[2]/l]; }
function sub3(a,b) { return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function cross3(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot3(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

// ─── MESH GENERATION ─────────────────────────────────────
function makeSphere(stacks=64, slices=64) {
  const pos=[], nor=[], uv=[], tan=[], idx=[];
  for (let i=0;i<=stacks;i++) {
    const phi=Math.PI*i/stacks;
    for (let j=0;j<=slices;j++) {
      const theta=2*Math.PI*j/slices;
      const x=Math.sin(phi)*Math.cos(theta);
      const y=Math.cos(phi);
      const z=Math.sin(phi)*Math.sin(theta);
      pos.push(x,y,z); nor.push(x,y,z);
      uv.push(j/slices, i/stacks);
      tan.push(-Math.sin(theta),0,Math.cos(theta));
    }
  }
  for (let i=0;i<stacks;i++) for(let j=0;j<slices;j++) {
    const a=i*(slices+1)+j, b=a+slices+1;
    idx.push(a,b,a+1, b,b+1,a+1);
  }
  return { pos, nor, uv, tan, idx, name:`SPHERE · ${stacks}×${slices}` };
}

function makeCube() {
  // 6 faces
  const faces=[
    { n:[0,0,1],  t:[1,0,0],  u:[0,1,0] },
    { n:[0,0,-1], t:[-1,0,0], u:[0,1,0] },
    { n:[1,0,0],  t:[0,0,-1], u:[0,1,0] },
    { n:[-1,0,0], t:[0,0,1],  u:[0,1,0] },
    { n:[0,1,0],  t:[1,0,0],  u:[0,0,-1] },
    { n:[0,-1,0], t:[1,0,0],  u:[0,0,1] },
  ];
  const pos=[],nor=[],uv=[],tan=[],idx=[];
  faces.forEach(({n,t,u},fi) => {
    const right=t, up=u, fwd=n;
    for(let i=0;i<4;i++){
      const s=(i%2===0)?-1:1;
      const q=(i<2)?-1:1;
      const v=[right[0]*s+fwd[0], right[1]*s+fwd[1], right[2]*s+fwd[2]];
      const w=[up[0]*q, up[1]*q, up[2]*q];
      pos.push(v[0]+w[0], v[1]+w[1], v[2]+w[2]);
      nor.push(...n); tan.push(...t);
      uv.push(i%2, Math.floor(i/2));
    }
    const base=fi*4;
    idx.push(base,base+1,base+2, base+1,base+3,base+2);
  });
  return { pos, nor, uv, tan, idx, name:'CUBE · 6 FACES' };
}

function makeTorus(R=0.7, r=0.3, segs=64, sides=32) {
  const pos=[],nor=[],uv=[],tan=[],idx=[];
  for(let i=0;i<=segs;i++){
    const u=2*Math.PI*i/segs;
    const cu=Math.cos(u), su=Math.sin(u);
    for(let j=0;j<=sides;j++){
      const v=2*Math.PI*j/sides;
      const cv=Math.cos(v), sv=Math.sin(v);
      const x=(R+r*cv)*cu, y=r*sv, z=(R+r*cv)*su;
      const nx=cv*cu, ny=sv, nz=cv*su;
      const tx=-su, ty=0, tz=cu;
      pos.push(x,y,z); nor.push(nx,ny,nz);
      uv.push(i/segs, j/sides);
      tan.push(tx,ty,tz);
    }
  }
  for(let i=0;i<segs;i++) for(let j=0;j<sides;j++){
    const a=i*(sides+1)+j, b=a+sides+1;
    idx.push(a,b,a+1, b,b+1,a+1);
  }
  return { pos, nor, uv, tan, idx, name:`TORUS · ${segs}×${sides}` };
}

function makeCylinder(segs=64, stacks=32) {
  const pos=[],nor=[],uv=[],tan=[],idx=[];
  // side
  for(let i=0;i<=stacks;i++){
    const y=-1+2*i/stacks;
    for(let j=0;j<=segs;j++){
      const a=2*Math.PI*j/segs;
      const x=Math.cos(a), z=Math.sin(a);
      pos.push(x,y,z); nor.push(x,0,z);
      uv.push(j/segs,i/stacks);
      tan.push(-Math.sin(a),0,Math.cos(a));
    }
  }
  for(let i=0;i<stacks;i++) for(let j=0;j<segs;j++){
    const a=i*(segs+1)+j, b=a+segs+1;
    idx.push(a,b,a+1, b,b+1,a+1);
  }
  // caps
  [[1,'top'],[-1,'bot']].forEach(([sy])=>{
    const base=pos.length/3;
    pos.push(0,sy,0); nor.push(0,sy,0); uv.push(.5,.5); tan.push(1,0,0);
    for(let j=0;j<=segs;j++){
      const a=2*Math.PI*j/segs;
      pos.push(Math.cos(a),sy,Math.sin(a));
      nor.push(0,sy,0); tan.push(1,0,0);
      uv.push(Math.cos(a)*.5+.5, Math.sin(a)*.5+.5);
    }
    for(let j=0;j<segs;j++){
      if(sy>0) idx.push(base, base+j+1, base+j+2);
      else     idx.push(base, base+j+2, base+j+1);
    }
  });
  return { pos, nor, uv, tan, idx, name:`CYLINDER · ${segs}×${stacks}` };
}

// ─── GPU BUFFER UPLOAD ───────────────────────────────────
function uploadMesh(meshData) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  function buf(data, loc, size) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    return b;
  }

  buf(meshData.pos, 0, 3);
  buf(meshData.nor, 1, 3);
  buf(meshData.uv,  2, 2);
  buf(meshData.tan, 3, 3);

  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(meshData.idx), gl.STATIC_DRAW);

  gl.bindVertexArray(null);
  return { vao, count: meshData.idx.length, verts: meshData.pos.length/3, name: meshData.name };
}

// ─── UNIFORMS ────────────────────────────────────────────
function ul(name) { return gl.getUniformLocation(program, name); }
const locs = {};
function cacheLocs() {
  ['uModel','uView','uProj','uNormalMat',
   'uAlbedo','uMetallic','uRoughness','uAO','uEmissive',
   'uCamPos','uAmbient',
   'uTonemap','uGamma','uExposure','uChannel',
   'uHeightMap','uParallaxScale','uParallaxSteps','uParallaxEnabled','uParallaxShadow'].forEach(n => { locs[n] = ul(n); });
  for(let i=0;i<3;i++){
    locs[`uLightPos${i}`]       = ul(`uLightPos[${i}]`);
    locs[`uLightColor${i}`]     = ul(`uLightColor[${i}]`);
    locs[`uLightIntensity${i}`] = ul(`uLightIntensity[${i}]`);
  }
}

// ─── RENDER ──────────────────────────────────────────────
const mModel = mat4Identity(mat4());
const mView  = mat4();
const mProj  = mat4();
const mMV    = mat4();
const mRot   = mat4();

function getEye() {
  const {theta, phi, radius, panX, panY} = camera;
  return [
    radius * Math.sin(phi) * Math.cos(theta) + panX,
    radius * Math.cos(phi) + panY,
    radius * Math.sin(phi) * Math.sin(theta),
  ];
}

function render(ts) {
  // FPS
  frameCount++;
  const elapsed = ts - lastFpsTime;
  if (elapsed >= 500) {
    fps = Math.round(frameCount / (elapsed/1000));
    frameCount = 0; lastFpsTime = ts;
    document.getElementById('fps-counter').textContent = `${fps} FPS`;
  }

  // Resize
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
    gl.viewport(0, 0, w, h);
  }

//  gl.clearColor(0.04, 0.043, 0.05, 1);
  gl.clearColor(0.9804, 0.9804, 0.9804, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  gl.useProgram(program);
  const current = meshBuffers[state.mesh];
  if (!current) { requestAnimationFrame(render); return; }

  // Auto rotate
  if (state.autoRotate) {
    autoRotAngle += state.rotSpeed * 0.008;
    mat4Rotate(mRot, autoRotAngle, [0, 1, 0]);
  } else {
    mat4Identity(mRot);
  }

  const eye = getEye();
  mat4LookAt(mView, eye, [camera.panX, camera.panY, 0], [0,1,0]);
  mat4Perspective(mProj, Math.PI/4, canvas.width/canvas.height, 0.1, 100);
  const normalMat = mat3NormalFromMat4(mRot);

  gl.uniformMatrix4fv(locs.uModel, false, mRot);
  gl.uniformMatrix4fv(locs.uView,  false, mView);
  gl.uniformMatrix4fv(locs.uProj,  false, mProj);
  gl.uniformMatrix3fv(locs.uNormalMat, false, normalMat);

  gl.uniform3fv(locs.uAlbedo,   state.albedo);
  gl.uniform1f(locs.uMetallic,  state.metallic);
  gl.uniform1f(locs.uRoughness, state.roughness);
  gl.uniform1f(locs.uAO,        state.ao);
  gl.uniform1f(locs.uEmissive,  state.emissive);
  gl.uniform3fv(locs.uCamPos,   eye);
  gl.uniform3fv(locs.uAmbient,  state.ambientColor);

  for(let i=0;i<3;i++){
    gl.uniform3fv(locs[`uLightPos${i}`],       state.lights[i].pos);
    gl.uniform3fv(locs[`uLightColor${i}`],     state.lights[i].color);
    gl.uniform1f(locs[`uLightIntensity${i}`],  state.lights[i].intensity);
  }

  gl.uniform1i(locs.uTonemap,  state.tonemapping ? 1 : 0);
  gl.uniform1i(locs.uGamma,    state.gamma ? 1 : 0);
  gl.uniform1f(locs.uExposure, state.exposure);
  gl.uniform1i(locs.uChannel,  state.channel);

  // Parallax
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, heightTex);
  gl.uniform1i(locs.uHeightMap,        0);
  gl.uniform1f(locs.uParallaxScale,    state.parallaxEnabled ? state.parallaxScale : 0.0);
  gl.uniform1i(locs.uParallaxSteps,    state.parallaxSteps);
  gl.uniform1i(locs.uParallaxEnabled,  state.parallaxEnabled ? 1 : 0);
  gl.uniform1i(locs.uParallaxShadow,   state.parallaxShadow ? 1 : 0);

  gl.bindVertexArray(current.vao);
  gl.drawElements(gl.TRIANGLES, current.count, gl.UNSIGNED_INT, 0);
  gl.bindVertexArray(null);

  requestAnimationFrame(render);
}

// ─── MOUSE / ORBIT ───────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  if (e.button === 0) mouse.down = true;
  if (e.button === 2) mouse.rightDown = true;
  mouse.x = e.clientX; mouse.y = e.clientY;
});

window.addEventListener('mouseup', e => {
  if (e.button === 0) mouse.down = false;
  if (e.button === 2) mouse.rightDown = false;
});

window.addEventListener('mousemove', e => {
  const dx = e.clientX - mouse.x;
  const dy = e.clientY - mouse.y;
  mouse.x = e.clientX; mouse.y = e.clientY;
  if (mouse.down) {
    camera.theta -= dx * 0.007;
    camera.phi   = Math.max(0.1, Math.min(Math.PI-0.1, camera.phi + dy * 0.007));
  }
  if (mouse.rightDown) {
    camera.panX -= dx * 0.003;
    camera.panY += dy * 0.003;
  }
});

canvas.addEventListener('wheel', e => {
  camera.radius = Math.max(0.8, Math.min(10, camera.radius + e.deltaY * 0.005));
  e.preventDefault();
}, { passive: false });

canvas.addEventListener('contextmenu', e => e.preventDefault());

// ─── UI CALLBACKS ────────────────────────────────────────
function hexToRGB(hex) {
  const r = parseInt(hex.slice(1,3),16)/255;
  const g = parseInt(hex.slice(3,5),16)/255;
  const b = parseInt(hex.slice(5,7),16)/255;
  return [r,g,b];
}

window.setMesh = function(name) {
  state.mesh = name;
  document.querySelectorAll('.mesh-btn').forEach(b => b.classList.toggle('active', b.textContent.toLowerCase()===name));
  const m = meshBuffers[name];
  if (m) {
    document.getElementById('stat-verts').textContent = m.verts.toLocaleString();
    document.getElementById('stat-tris').textContent = (m.count/3).toLocaleString();
    document.getElementById('sb-mesh').textContent = m.name;
  }
};

const presets = {
  gold:    { albedo:'#ffd700', metallic:1.0, roughness:0.18, ao:1.0 },
  silver:  { albedo:'#d0d0d5', metallic:1.0, roughness:0.12, ao:1.0 },
  copper:  { albedo:'#b87333', metallic:1.0, roughness:0.25, ao:1.0 },
  iron:    { albedo:'#888888', metallic:0.9, roughness:0.55, ao:0.9 },
  plastic: { albedo:'#2244cc', metallic:0.0, roughness:0.4,  ao:1.0 },
  rubber:  { albedo:'#1a1a1a', metallic:0.0, roughness:0.95, ao:0.8 },
  obsidian:{ albedo:'#1a1a2e', metallic:0.0, roughness:0.05, ao:1.0 },
  jade:    { albedo:'#4aa96c', metallic:0.0, roughness:0.3,  ao:0.9 },
};

window.applyPreset = function(name) {
  const p = presets[name];
  if (!p) return;
  state.albedo    = hexToRGB(p.albedo);
  state.metallic  = p.metallic;
  state.roughness = p.roughness;
  state.ao        = p.ao;
  // Update UI
  document.getElementById('albedo-color').value       = p.albedo;
  document.getElementById('sl-metallic').value        = p.metallic;
  document.getElementById('val-metallic').textContent = p.metallic.toFixed(2);
  document.getElementById('sl-roughness').value       = p.roughness;
  document.getElementById('val-roughness').textContent= p.roughness.toFixed(2);
  document.getElementById('sl-ao').value              = p.ao;
  document.getElementById('val-ao').textContent       = p.ao.toFixed(2);
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.toggle('active', b.getAttribute('onclick').includes(name)));
  document.getElementById('sb-preset').textContent = name.toUpperCase()+' PRESET';
};

window.updateAlbedo = function(hex) {
  state.albedo = hexToRGB(hex);
};

window.updateParam = function(param, val) {
  state[param] = parseFloat(val);
  document.getElementById(`val-${param}`).textContent = parseFloat(val).toFixed(2);
};

window.updateLight = function(idx, prop, val) {
  if (prop === 'intensity') {
    state.lights[idx].intensity = parseFloat(val);
    document.getElementById(`val-light${idx}`).textContent = parseFloat(val).toFixed(2);
  } else if (prop === 'color') {
    state.lights[idx].color = hexToRGB(val);
  }
};

const envPresets = {
  studio:  { ambient:[0.08,0.09,0.12], lights:[[3,4,3],[-4,2,-2],[0,-3,4]] },
  sunset:  { ambient:[0.15,0.08,0.04] },
  overcast:{ ambient:[0.18,0.18,0.20] },
  night:   { ambient:[0.02,0.02,0.04] },
};

window.setEnv = function(name) {
  document.querySelectorAll('.light-btn').forEach(b => b.classList.toggle('active', b.textContent.trim().toLowerCase().includes(name)));
  document.getElementById('sb-env').textContent = name.toUpperCase()+' ENV';
  const env = envPresets[name] || {};
  if (env.ambient) state.ambientColor = env.ambient;
  if (name==='sunset') {
    state.lights[0].color=[1.0,0.7,0.4]; state.lights[0].intensity=5.0;
    state.lights[1].color=[0.4,0.3,0.8]; state.lights[1].intensity=1.0;
    document.getElementById('lc0').value='#ffb366'; document.getElementById('sl-light0').value=5.0; document.getElementById('val-light0').textContent='5.00';
  } else if (name==='studio') {
    state.lights[0].color=[1,1,1]; state.lights[0].intensity=3.0;
    state.lights[1].color=[0.53,0.67,1.0]; state.lights[1].intensity=1.5;
    document.getElementById('lc0').value='#ffffff'; document.getElementById('sl-light0').value=3.0; document.getElementById('val-light0').textContent='3.00';
  } else if (name==='night') {
    state.lights[0].color=[0.3,0.4,1.0]; state.lights[0].intensity=0.5;
    state.lights[1].color=[0.1,0.1,0.2]; state.lights[1].intensity=0.2;
    document.getElementById('sl-light0').value=0.5; document.getElementById('val-light0').textContent='0.50';
  } else if (name==='overcast') {
    state.lights[0].color=[0.8,0.85,0.9]; state.lights[0].intensity=2.0;
    state.lights[1].color=[0.75,0.8,0.85]; state.lights[1].intensity=1.8;
  }
};

window.toggleTonemap = val => { state.tonemapping = val; };
window.toggleGamma   = val => { state.gamma = val; };
window.updateExposure = val => {
  state.exposure = parseFloat(val);
  document.getElementById('val-exposure').textContent = parseFloat(val).toFixed(2);
};
window.toggleRotate = val => { state.autoRotate = val; };
window.updateRotSpeed = val => {
  state.rotSpeed = parseFloat(val);
  document.getElementById('val-rotspeed').textContent = parseFloat(val).toFixed(2);
};

const channelNames = ['FULL PBR','ALBEDO','NORMALS','METALLIC','ROUGHNESS','AO','HEIGHT'];
window.setChannel = function(ch) {
  state.channel = ch;
  document.querySelectorAll('.channel-btn').forEach((b,i) => b.classList.toggle('active', i===ch));
  document.getElementById('sb-channel').textContent = channelNames[ch];
};

// ─── HEIGHT MAP GENERATORS ───────────────────────────────
function generateHeightTexture(pattern) {
  const SIZE = 512;
  const data = new Uint8Array(SIZE * SIZE);

  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const u = x / SIZE, v = y / SIZE;
      let h = 0;

      if (pattern === 'bricks') {
        const bricksX = 8, bricksY = 4;
        const row = Math.floor(v * bricksY);
        const offset = (row % 2) * 0.5;
        const bx = ((u + offset) * bricksX) % 1.0;
        const by = (v * bricksY) % 1.0;
        const mortarW = 0.06;
        const inBrick = (bx > mortarW && bx < 1.0-mortarW && by > mortarW && by < 1.0-mortarW);
        if (inBrick) {
          // Slight height variation per brick + small noise
          const brickId = Math.floor((u + offset) * bricksX) + row * bricksX;
          const noise = Math.sin(brickId * 127.1 + 311.7) * 0.5 + 0.5;
          const cx = Math.abs(bx - 0.5) * 2.0, cy = Math.abs(by - 0.5) * 2.0;
          const edge = 1.0 - Math.max(cx, cy);
          h = 0.6 + noise * 0.15 + edge * 0.15;
        } else {
          h = 0.05;
        }

      } else if (pattern === 'tiles') {
        const n = 6;
        const tx = (u * n) % 1.0, ty = (v * n) % 1.0;
        const grout = 0.05;
        const inTile = tx > grout && tx < 1-grout && ty > grout && ty < 1-grout;
        if (inTile) {
          const cx = Math.abs(tx - 0.5) * 2, cy = Math.abs(ty - 0.5) * 2;
          h = 0.7 + (1.0 - Math.max(cx, cy)) * 0.2;
        } else {
          h = 0.0;
        }

      } else if (pattern === 'waves') {
        const freq1 = 8, freq2 = 12, freq3 = 5;
        h = 0.5
          + Math.sin(u * freq1 * Math.PI * 2) * 0.18
          + Math.sin(v * freq2 * Math.PI * 2) * 0.14
          + Math.sin((u + v) * freq3 * Math.PI * 2) * 0.10
          + Math.sin(u * 20 + v * 15) * 0.05;
        h = Math.max(0, Math.min(1, h));

      } else if (pattern === 'cobble') {
        // Voronoi-ish cobblestones
        let minDist = 1e9;
        for (let i = -1; i <= 1; i++) {
          for (let j = -1; j <= 1; j++) {
            const cellX = Math.floor(u * 6) + i;
            const cellY = Math.floor(v * 6) + j;
            const seed = (cellX * 127 + cellY * 311) & 0xffff;
            const px = (cellX + 0.3 + (Math.sin(seed * 1.7) * 0.5 + 0.5) * 0.55) / 6;
            const py = (cellY + 0.3 + (Math.cos(seed * 2.3) * 0.5 + 0.5) * 0.55) / 6;
            const dx = u - px, dy = v - py;
            minDist = Math.min(minDist, dx*dx + dy*dy);
          }
        }
        const d = Math.sqrt(minDist) * 6;
        h = d < 0.38 ? 0.0 : Math.pow(Math.min(1, (d - 0.38) / 0.28), 0.5) * 0.9;

      } else if (pattern === 'scratches') {
        // Base + procedural scratches
        h = 0.7;
        for (let s = 0; s < 24; s++) {
          const sx = Math.sin(s * 13.7) * 0.5 + 0.5;
          const sy = Math.cos(s * 7.3) * 0.5 + 0.5;
          const ang = Math.sin(s * 4.1) * Math.PI;
          const dx = Math.cos(ang), dy = Math.sin(ang);
          const t2 = (u - sx) * dx + (v - sy) * dy;
          const perp = Math.abs((u - sx) * dy - (v - sy) * dx);
          if (perp < 0.003 && t2 > -0.2 && t2 < 0.2) {
            h -= 0.45 * (1.0 - perp / 0.003) * (1.0 - Math.abs(t2) * 5.0);
          }
        }
        h = Math.max(0, Math.min(1, h));

      } else { // noise
        // Multi-octave FBM
        let amp = 0.5, freq2 = 4, val = 0;
        for (let oct = 0; oct < 5; oct++) {
          val += Math.sin(u * freq2 * Math.PI * 2 + oct * 1.7) *
                 Math.cos(v * freq2 * Math.PI * 2 + oct * 2.3) * amp;
          freq2 *= 2.1; amp *= 0.48;
        }
        h = val * 0.5 + 0.5;
      }

      data[y * SIZE + x] = Math.round(Math.max(0, Math.min(1, h)) * 255);
    }
  }

  if (!heightTex) heightTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, heightTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, SIZE, SIZE, 0, gl.RED, gl.UNSIGNED_BYTE, data);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
}

window.setHeightPattern = function(p) {
  state.heightPattern = p;
  generateHeightTexture(p);
  document.querySelectorAll('.hmap-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.pattern === p));
};

window.toggleParallax = val => { state.parallaxEnabled = val; };
window.toggleParallaxShadow = val => { state.parallaxShadow = val; };
window.updateParallaxScale = val => {
  state.parallaxScale = parseFloat(val);
  document.getElementById('val-pscale').textContent = parseFloat(val).toFixed(3);
};
window.updateParallaxSteps = val => {
  state.parallaxSteps = parseInt(val);
  document.getElementById('val-psteps').textContent = val;
};

// ─── INIT ────────────────────────────────────────────────
function init() {
  program = createProgram(VS, FS);
  cacheLocs();

  generateHeightTexture(state.heightPattern);

  meshBuffers.sphere   = uploadMesh(makeSphere(64,64));
  meshBuffers.cube     = uploadMesh(makeCube());
  meshBuffers.torus    = uploadMesh(makeTorus());
  meshBuffers.cylinder = uploadMesh(makeCylinder());

  setMesh('sphere');
  requestAnimationFrame(render);
}

init();

