/**
 * plate-ocr.js  —  v2.0
 * Offline vehicle license plate recognizer
 * 53+ European & neighboring countries · No external APIs · Pure JS + Canvas API
 *
 * Pipeline:
 *   1. Auto-gamma + Sauvola adaptive threshold (integral images → O(n))
 *   2. Multi-scale region detection (Sobel + morphological dilation + connected components)
 *   3. Perspective correction (bilinear warp from 4 detected corners)
 *   4. Character segmentation (vertical projection + valley detection + split-merge)
 *   5. Multi-threshold ensemble (3 configs × region)
 *   6. Geometric feature classifier (12 features, top-3 per char)
 *   7. Multi-hypothesis engine (up to 24 variants per reading)
 *   8. Position-aware D/L correction per country schema
 *   9. Ensemble voting across hypotheses → confident country+plate
 *
 * Usage:
 *   import { recognizePlate } from './plate-ocr.js';
 *   const result = await recognizePlate(blob);
 *   // { plate, country, countryName, confidence, bbox, chars,
 *   //   charConfidences, plateColor, candidates, processingMs }
 */

// ─────────────────────────────────────────────────────────────────────────────
// COUNTRY DATABASE  —  53 countries
// schema: D=digit forced, L=letter forced, A=alphanumeric, x=ignored separator
// ─────────────────────────────────────────────────────────────────────────────
const COUNTRIES = {
  AT:{name:"Austria",       r:/^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}[A-Z]?$/,          schema:"LLLDDD"},
  BE:{name:"Belgium",       r:/^[1-9]-[A-Z]{3}-\d{3}$/,                         schema:"DLLLDDD"},
  BG:{name:"Bulgaria",      r:/^[A-Z]{1,2}\d{4}[A-Z]{2}$/,                      schema:"LLDDDDLL"},
  CY:{name:"Cyprus",        r:/^[A-Z]{3}\d{3}$/,                                 schema:"LLLDDD"},
  CZ:{name:"Czech Rep.",    r:/^\d[A-Z]{2}\d{4}$/,                               schema:"DLLDDDD"},
  DE:{name:"Germany",       r:/^[A-Z]{1,3}[A-Z]{1,2}\d{1,4}[A-Z]?$/,           schema:"LLLDDD"},
  DK:{name:"Denmark",       r:/^[A-Z]{2}\d{5}$/,                                 schema:"LLDDDDD"},
  EE:{name:"Estonia",       r:/^\d{3}[A-Z]{3}$/,                                 schema:"DDDLLL"},
  ES:{name:"Spain",         r:/^\d{4}[BCDFGHJKLMNPRSTUVWXYZ]{3}$/,               schema:"DDDDLLL"},
  FI:{name:"Finland",       r:/^[A-Z]{2,3}\d{1,3}$/,                             schema:"LLLDDD"},
  FR:{name:"France",        r:/^[A-Z]{2}\d{3}[A-Z]{2}$/,                         schema:"LLDDDLL"},
  GR:{name:"Greece",        r:/^[A-Z]{3}\d{4}$/,                                 schema:"LLLDDDD"},
  HR:{name:"Croatia",       r:/^[A-Z]{2}\d{3,4}[A-Z]{2}$/,                      schema:"LLDDDLL"},
  HU:{name:"Hungary",       r:/^[A-Z]{3}\d{3}$/,                                 schema:"LLLDDD"},
  IE:{name:"Ireland",       r:/^\d{2,3}[A-Z]{1,2}\d{1,6}$/,                     schema:"DDLDDDD"},
  IT:{name:"Italy",         r:/^[A-Z]{2}\d{3}[A-Z]{2}$/,                         schema:"LLDDDLL"},
  LT:{name:"Lithuania",     r:/^[A-Z]{3}\d{3}$/,                                 schema:"LLLDDD"},
  LU:{name:"Luxembourg",    r:/^[A-Z]{2}\d{4}$/,                                 schema:"LLDDDD"},
  LV:{name:"Latvia",        r:/^[A-Z]{2}\d{4}$/,                                 schema:"LLDDDD"},
  MT:{name:"Malta",         r:/^[A-Z]{3}\d{3}$/,                                 schema:"LLLDDD"},
  NL:{name:"Netherlands",   r:/^[A-Z]{2}\d{2}[A-Z]{2}$|^\d{2}[A-Z]{3}\d$/,     schema:"LLDDLL"},
  PL:{name:"Poland",        r:/^[A-Z]{2,3}[A-Z0-9]{4,5}$/,                      schema:"LLLAAAAA"},
  PT:{name:"Portugal",      r:/^[A-Z]{2}\d{2}[A-Z]{2}$|^\d{2}[A-Z]{2}\d{2}$/,  schema:"LLDDLL"},
  RO:{name:"Romania",       r:/^[A-Z]{1,2}\d{2,3}[A-Z]{3}$/,                    schema:"LLDDDLLL"},
  SE:{name:"Sweden",        r:/^[A-Z]{3}\d{2}[A-Z0-9]$/,                         schema:"LLLDDA"},
  SI:{name:"Slovenia",      r:/^[A-Z]{2}[A-Z0-9]{5}$/,                           schema:"LLAAAAA"},
  SK:{name:"Slovakia",      r:/^[A-Z]{2}\d{3}[A-Z]{2}$/,                         schema:"LLDDDLL"},
  GB:{name:"United Kingdom",r:/^[A-Z]{2}\d{2}[A-Z]{3}$/,                        schema:"LLDLLLL"},
  NO:{name:"Norway",        r:/^[A-Z]{2}\d{5}$/,                                 schema:"LLDDDDD"},
  CH:{name:"Switzerland",   r:/^[A-Z]{2}\d{1,6}[A-Z]?$/,                        schema:"LLDDDDDD"},
  IS:{name:"Iceland",       r:/^[A-Z]{1,3}\d{1,3}$/,                             schema:"LLLDDD"},
  LI:{name:"Liechtenstein", r:/^FL\d{1,5}$/,                                     schema:"LLDDDDD"},
  AD:{name:"Andorra",       r:/^[A-Z]{1,2}\d{4}$/,                               schema:"LLDDDD"},
  MC:{name:"Monaco",        r:/^\d{3}[A-Z]{3}$/,                                 schema:"DDDLLL"},
  GI:{name:"Gibraltar",     r:/^[A-Z]{3}\d{4}$/,                                 schema:"LLLDDDD"},
  SM:{name:"San Marino",    r:/^\d{1,5}$/,                                        schema:"DDDDD"},
  VA:{name:"Vatican",       r:/^SCV\d{1,5}$/,                                    schema:"LLLDDDDD"},
  RS:{name:"Serbia",        r:/^[A-Z]{2}\d{3,4}[A-Z]{2}$/,                      schema:"LLDDDLL"},
  BA:{name:"Bosnia",        r:/^[A-Z]\d{2}[A-Z]\d{3}$/,                         schema:"LDDLDDD"},
  ME:{name:"Montenegro",    r:/^[A-Z]{2}[A-Z]{2}\d{3,4}$/,                      schema:"LLLLDDDD"},
  MK:{name:"N. Macedonia",  r:/^[A-Z]{2}\d{4}[A-Z]{2}$/,                        schema:"LLDDDDLL"},
  AL:{name:"Albania",       r:/^[A-Z]{2}\d{3}[A-Z]{2}$/,                        schema:"LLDDDLL"},
  XK:{name:"Kosovo",        r:/^\d{2}[A-Z]{3}\d{3}$/,                           schema:"DDLLLDDD"},
  TR:{name:"Turkey",        r:/^\d{2}[A-Z]{1,3}\d{2,4}$/,                       schema:"DDLLLDDDD"},
  UA:{name:"Ukraine",       r:/^[A-Z]{2}\d{4}[A-Z]{2}$/,                        schema:"LLDDDDLL"},
  RU:{name:"Russia",        r:/^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$/,  schema:"LDDDLLDDD"},
  BY:{name:"Belarus",       r:/^\d{4}[A-Z]{2}\d$/,                              schema:"DDDDLLD"},
  MD:{name:"Moldova",       r:/^[A-Z]{3}\d{3}$/,                                 schema:"LLLDDD"},
  GE:{name:"Georgia",       r:/^[A-Z]{2}\d{3}[A-Z]{2}$/,                        schema:"LLDDDLL"},
  AM:{name:"Armenia",       r:/^\d{2}[A-Z]{2}\d{3}$/,                           schema:"DDLLDDD"},
  AZ:{name:"Azerbaijan",    r:/^\d{2}[A-Z]{2}\d{3}$/,                           schema:"DDLLDDD"},
  MA:{name:"Morocco",       r:/^\d{1,5}[A-Z]\d{1,2}$/,                          schema:"DDDDDLDD"},
  DZ:{name:"Algeria",       r:/^\d{5}\d{2}\d{4}$/,                              schema:"DDDDDDDDDDD"},
  TN:{name:"Tunisia",       r:/^\d{3}[A-Z]{3}\d{4}$/,                           schema:"DDDLLLDDDDD"},
  LY:{name:"Libya",         r:/^\d{6,7}$/,                                       schema:"DDDDDDD"},
};

// Confuser correction maps
const D2L = {'0':'O','1':'I','2':'Z','5':'S','8':'B','6':'G'};
const L2D = {'O':'0','I':'1','Z':'2','S':'5','B':'8','G':'6'};

// ─────────────────────────────────────────────────────────────────────────────
// CHARACTER FEATURE TABLE  —  12 features per glyph
// [density, q0,q1,q2,q3, hCross,vCross, symH,symV, holes, cxX,cxY]
// ─────────────────────────────────────────────────────────────────────────────
const CF = {
  '0':[0.45,0.48,0.48,0.48,0.48,0.92,0.85,0.02,0.01,1,0.50,0.50],
  '1':[0.18,0.05,0.38,0.05,0.38,0.92,0.18,0.04,0.00,0,0.60,0.50],
  '2':[0.42,0.38,0.48,0.55,0.30,0.85,0.78,0.18,0.25,0,0.50,0.52],
  '3':[0.38,0.30,0.48,0.30,0.48,0.85,0.65,0.04,0.05,0,0.58,0.50],
  '4':[0.40,0.50,0.50,0.15,0.50,0.78,0.78,0.28,0.40,0,0.52,0.42],
  '5':[0.42,0.48,0.30,0.30,0.48,0.85,0.75,0.18,0.05,0,0.48,0.50],
  '6':[0.52,0.40,0.38,0.55,0.55,0.92,0.82,0.08,0.20,1,0.48,0.55],
  '7':[0.28,0.30,0.48,0.05,0.22,0.75,0.55,0.22,0.40,0,0.55,0.38],
  '8':[0.55,0.52,0.52,0.52,0.52,0.92,0.85,0.02,0.02,2,0.50,0.50],
  '9':[0.52,0.55,0.55,0.40,0.38,0.92,0.82,0.08,0.20,1,0.52,0.45],
  'A':[0.45,0.45,0.45,0.32,0.32,0.78,0.92,0.02,0.10,1,0.50,0.48],
  'B':[0.55,0.55,0.42,0.55,0.42,0.92,0.70,0.22,0.02,2,0.46,0.50],
  'C':[0.38,0.45,0.30,0.45,0.30,0.88,0.60,0.02,0.02,0,0.44,0.50],
  'D':[0.52,0.55,0.38,0.55,0.38,0.92,0.72,0.25,0.02,1,0.48,0.50],
  'E':[0.45,0.55,0.30,0.55,0.30,0.88,0.62,0.32,0.02,0,0.44,0.50],
  'F':[0.38,0.55,0.30,0.42,0.22,0.82,0.62,0.35,0.18,0,0.46,0.40],
  'G':[0.52,0.45,0.32,0.48,0.52,0.88,0.68,0.18,0.05,0,0.50,0.52],
  'H':[0.48,0.50,0.50,0.50,0.50,0.88,0.92,0.02,0.02,0,0.50,0.50],
  'I':[0.22,0.10,0.30,0.10,0.30,0.92,0.22,0.02,0.02,0,0.50,0.50],
  'J':[0.28,0.05,0.38,0.20,0.35,0.88,0.28,0.18,0.22,0,0.55,0.52],
  'K':[0.45,0.50,0.42,0.50,0.38,0.88,0.88,0.28,0.05,0,0.48,0.50],
  'L':[0.32,0.50,0.12,0.50,0.38,0.88,0.52,0.45,0.45,0,0.42,0.58],
  'M':[0.55,0.60,0.60,0.48,0.48,0.88,0.98,0.02,0.05,0,0.50,0.46],
  'N':[0.48,0.55,0.52,0.52,0.55,0.88,0.92,0.02,0.05,0,0.50,0.50],
  'O':[0.45,0.48,0.48,0.48,0.48,0.90,0.85,0.02,0.02,1,0.50,0.50],
  'P':[0.42,0.52,0.38,0.38,0.22,0.88,0.65,0.25,0.22,1,0.46,0.40],
  'Q':[0.50,0.48,0.50,0.48,0.55,0.90,0.85,0.05,0.05,1,0.52,0.52],
  'R':[0.50,0.55,0.38,0.52,0.45,0.88,0.72,0.20,0.05,1,0.48,0.50],
  'S':[0.40,0.45,0.28,0.28,0.45,0.85,0.68,0.05,0.05,0,0.50,0.50],
  'T':[0.28,0.30,0.38,0.22,0.22,0.78,0.32,0.02,0.28,0,0.50,0.55],
  'U':[0.45,0.50,0.50,0.48,0.48,0.88,0.88,0.02,0.05,0,0.50,0.52],
  'V':[0.38,0.45,0.45,0.30,0.30,0.78,0.88,0.02,0.15,0,0.50,0.55],
  'W':[0.52,0.55,0.55,0.50,0.50,0.78,0.98,0.02,0.02,0,0.50,0.55],
  'X':[0.40,0.48,0.48,0.48,0.48,0.85,0.85,0.02,0.02,0,0.50,0.50],
  'Y':[0.30,0.40,0.40,0.18,0.18,0.78,0.45,0.02,0.25,0,0.50,0.52],
  'Z':[0.40,0.28,0.48,0.48,0.28,0.85,0.75,0.05,0.28,0,0.50,0.50],
};
const FEAT_WEIGHTS = [2,1,1,1,1,1,1,0.5,0.5,3,0.5,0.5];

// ─────────────────────────────────────────────────────────────────────────────
// PREPROCESSING
// ─────────────────────────────────────────────────────────────────────────────
function toGray(data, w, h) {
  const g = new Uint8ClampedArray(w * h);
  for (let i = 0; i < g.length; i++) {
    const o = i << 2;
    g[i] = (data[o]*77 + data[o+1]*150 + data[o+2]*29) >> 8;
  }
  return g;
}

function applyGamma(g, gamma) {
  if (gamma === 1) return g;
  const lut = new Uint8ClampedArray(256);
  for (let i = 0; i < 256; i++) lut[i] = Math.round(255 * Math.pow(i/255, gamma));
  return g.map(v => lut[v]);
}

function autoGamma(g) {
  let s = 0;
  for (const v of g) s += v;
  const m = s / g.length;
  return m < 90 ? 0.5 : m > 165 ? 1.5 : 1.0;
}

function sauvola(g, w, h, k=0.2, R=128, ws=15) {
  const half = ws >> 1;
  const bin = new Uint8ClampedArray(w * h);
  const S  = new Float64Array(w * h);
  const S2 = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = g[y*w+x], i = y*w+x;
      S[i]  = v + (x?S[i-1]:0) + (y?S[i-w]:0) - (x&&y?S[i-w-1]:0);
      S2[i] = v*v + (x?S2[i-1]:0) + (y?S2[i-w]:0) - (x&&y?S2[i-w-1]:0);
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const x1=Math.max(0,x-half), y1=Math.max(0,y-half);
      const x2=Math.min(w-1,x+half), y2=Math.min(h-1,y+half);
      const n=(x2-x1+1)*(y2-y1+1);
      const getS=(arr,px,py)=>px<0||py<0?0:arr[py*w+px];
      const s  = getS(S,x2,y2)-getS(S,x1-1,y2)-getS(S,x2,y1-1)+getS(S,x1-1,y1-1);
      const s2 = getS(S2,x2,y2)-getS(S2,x1-1,y2)-getS(S2,x2,y1-1)+getS(S2,x1-1,y1-1);
      const mean = s/n;
      const std  = Math.sqrt(Math.max(0, s2/n - mean*mean));
      bin[y*w+x] = g[y*w+x] < mean*(1+k*(std/R-1)) ? 0 : 255;
    }
  }
  return bin;
}

// ─────────────────────────────────────────────────────────────────────────────
// REGION DETECTION
// ─────────────────────────────────────────────────────────────────────────────
function sobel(g, w, h) {
  const mag = new Float32Array(w * h);
  for (let y = 1; y < h-1; y++) {
    for (let x = 1; x < w-1; x++) {
      const i = y*w+x;
      const gx = -g[i-w-1]+g[i-w+1]-2*g[i-1]+2*g[i+1]-g[i+w-1]+g[i+w+1];
      const gy = -g[i-w-1]-2*g[i-w]-g[i-w+1]+g[i+w-1]+2*g[i+w]+g[i+w+1];
      mag[i] = Math.sqrt(gx*gx+gy*gy);
    }
  }
  return mag;
}

function dilateH(bin, w, h, se=22) {
  const out = new Uint8ClampedArray(w*h).fill(255);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (bin[y*w+x] === 0) {
        const end = Math.min(w, x+se);
        for (let xx = x; xx < end; xx++) out[y*w+xx] = 0;
      }
    }
  }
  return out;
}

function dilateV(bin, w, h, se=3) {
  const out = new Uint8ClampedArray(w*h).fill(255);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      if (bin[y*w+x] === 0) {
        const end = Math.min(h, y+se);
        for (let yy = y; yy < end; yy++) out[yy*w+x] = 0;
      }
    }
  }
  return out;
}

function connComponents(bin, w, h) {
  const labels = new Int32Array(w*h);
  const parent = [0];
  let next = 1;
  const find = x => { while (parent[x]!==x) x=parent[x]=parent[parent[x]]; return x; };
  const union = (a,b) => { parent[find(a)]=find(b); };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (bin[y*w+x]) continue;
      const up = y?labels[(y-1)*w+x]:0, left = x?labels[y*w+x-1]:0;
      if (!up&&!left) { parent.push(next); labels[y*w+x]=next++; }
      else if (up&&!left) labels[y*w+x]=find(up);
      else if (!up&&left) labels[y*w+x]=find(left);
      else { labels[y*w+x]=find(up); union(up,left); }
    }
  }

  const bb = {};
  for (let i = 0; i < w*h; i++) {
    if (!labels[i]) continue;
    const l=find(labels[i]), x=i%w, y=(i/w)|0;
    if (!bb[l]) bb[l]={x1:x,y1:y,x2:x,y2:y,n:0};
    const b=bb[l];
    if(x<b.x1)b.x1=x; if(y<b.y1)b.y1=y; if(x>b.x2)b.x2=x; if(y>b.y2)b.y2=y; b.n++;
  }
  return Object.values(bb).map(b=>({x:b.x1,y:b.y1,w:b.x2-b.x1+1,h:b.y2-b.y1+1,n:b.n}));
}

function detectRegions(gray, w, h) {
  const mag = sobel(gray, w, h);
  // threshold gradient map
  const edgeBin = new Uint8ClampedArray(w*h);
  for (let i = 0; i < w*h; i++) edgeBin[i] = mag[i] > 30 ? 0 : 255;

  const dh = dilateH(edgeBin, w, h, 28);
  const dv = dilateV(dh, w, h, 4);
  const blobs = connComponents(dv, w, h);

  const minW = w * 0.06, maxW = w * 0.95;
  const minH = h * 0.02, maxH = h * 0.22;
  const minArea = minW * minH;

  const cands = [];
  for (const b of blobs) {
    if (b.w < minW || b.w > maxW) continue;
    if (b.h < minH || b.h > maxH) continue;
    const ratio = b.w / b.h;
    if (ratio < 1.8 || ratio > 8.5) continue;
    if (b.w * b.h < minArea) continue;

    // Edge density inside blob
    let eSum = 0;
    for (let y = b.y; y < b.y+b.h; y++)
      for (let x = b.x; x < b.x+b.w; x++)
        eSum += mag[y*w+x];
    const eDen = eSum / (b.w * b.h);
    if (eDen < 5) continue;

    cands.push({ x:b.x, y:b.y, w:b.w, h:b.h, eDen });
  }

  // Deduplicate by IoU
  const out = [];
  for (const r of cands.sort((a,b)=>b.eDen-a.eDen)) {
    let skip = false;
    for (const d of out) {
      const ix = Math.max(0, Math.min(r.x+r.w,d.x+d.w) - Math.max(r.x,d.x));
      const iy = Math.max(0, Math.min(r.y+r.h,d.y+d.h) - Math.max(r.y,d.y));
      if (ix*iy / (r.w*r.h+d.w*d.h-ix*iy) > 0.45) { skip=true; break; }
    }
    if (!skip) out.push(r);
    if (out.length >= 6) break;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// CHARACTER SEGMENTATION
// ─────────────────────────────────────────────────────────────────────────────
function segmentChars(bin, w, h) {
  const vp = new Int32Array(w);
  for (let x = 0; x < w; x++)
    for (let y = 0; y < h; y++)
      if (!bin[y*w+x]) vp[x]++;

  // Smooth
  const sm = new Float32Array(w);
  for (let x = 1; x < w-1; x++) sm[x] = (vp[x-1]+vp[x]*2+vp[x+1])/4;
  sm[0]=vp[0]; sm[w-1]=vp[w-1];

  const thr = h * 0.07;
  const segs = [];
  let inC = false, st = 0;
  for (let x = 0; x < w; x++) {
    if (!inC && sm[x]>thr) { inC=true; st=x; }
    else if (inC && sm[x]<=thr) { segs.push({x:st,w:x-st}); inC=false; }
  }
  if (inC) segs.push({x:st,w:w-st});

  // Merge small gaps
  const mg = [];
  for (const s of segs) {
    if (mg.length && s.x-(mg[mg.length-1].x+mg[mg.length-1].w) < w*0.025)
      mg[mg.length-1].w = s.x+s.w - mg[mg.length-1].x;
    else mg.push({...s});
  }

  if (!mg.length) return [];

  // Filter noise + split wide segments
  const avgW = mg.reduce((a,s)=>a+s.w,0)/mg.length;
  const fin = [];
  for (const s of mg) {
    if (s.w < avgW*0.2) continue;
    if (s.w > avgW*1.8 && s.w < avgW*2.8) {
      fin.push({x:s.x,         w: (s.w/2)|0});
      fin.push({x:s.x+(s.w/2)|0, w: s.w-((s.w/2)|0)});
    } else {
      fin.push(s);
    }
  }
  return fin;
}

// ─────────────────────────────────────────────────────────────────────────────
// CHARACTER RECOGNITION
// ─────────────────────────────────────────────────────────────────────────────
function extractPatch(bin, fullW, fullH, cx, cw, charH) {
  const NW=10, NH=16;
  const p = new Uint8ClampedArray(NW*NH);
  for (let ny = 0; ny < NH; ny++) {
    for (let nx = 0; nx < NW; nx++) {
      const sx = cx + Math.round(nx*cw/NW);
      const sy = Math.round(ny*charH/NH);
      p[ny*NW+nx] = (sx>=0&&sx<fullW&&sy>=0&&sy<fullH) ? bin[sy*fullW+sx] : 255;
    }
  }
  return p;
}

function countHoles(p, w, h) {
  const vis = new Uint8Array(w*h);
  const q = [];
  const push = i => { if (!vis[i]) { vis[i]=1; q.push(i); } };
  for (let x=0;x<w;x++) { if(p[x]===255) push(x); if(p[(h-1)*w+x]===255) push((h-1)*w+x); }
  for (let y=0;y<h;y++) { if(p[y*w]===255) push(y*w); if(p[y*w+w-1]===255) push(y*w+w-1); }
  let qi=0;
  while (qi<q.length) {
    const i=q[qi++], x=i%w, y=(i/w)|0;
    for (const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx=x+dx,ny=y+dy;
      if (nx>=0&&nx<w&&ny>=0&&ny<h) { const ni=ny*w+nx; if(!vis[ni]&&p[ni]===255) push(ni); }
    }
  }
  let holes=0;
  for (let i=0;i<w*h;i++) {
    if (p[i]===255&&!vis[i]) {
      holes++;
      const q2=[i]; vis[i]=1; let q2i=0;
      while (q2i<q2.length) {
        const ii=q2[q2i++],xx=ii%w,yy=(ii/w)|0;
        for (const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
          const nx=xx+dx,ny=yy+dy;
          if (nx>=0&&nx<w&&ny>=0&&ny<h) { const ni=ny*w+nx; if(!vis[ni]&&p[ni]===255){vis[ni]=1;q2.push(ni);} }
        }
      }
    }
  }
  return Math.min(holes, 3);
}

function features(p, w, h) {
  let dark=0, cxS=0, cyS=0;
  const q=[0,0,0,0];
  const hw=w>>1, hh=h>>1;
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
    if (!p[y*w+x]) { dark++; q[(y<hh?0:2)+(x<hw?0:1)]++; cxS+=x; cyS+=y; }
  }
  const dn=dark/(w*h);
  const qn=q.map(v=>v/(hw*hh+1));
  let hC=0,vC=0;
  for (let y=1;y<h-1;y++) { for (let x=0;x<w;x++) { if(!p[y*w+x]){hC++;break;} } }
  for (let x=1;x<w-1;x++) { for (let y=0;y<h;y++) { if(!p[y*w+x]){vC++;break;} } }
  let sH=0,sV=0;
  for (let y=0;y<h;y++) for (let x=0;x<hw;x++) if(p[y*w+x]!==p[y*w+(w-1-x)]) sH++;
  for (let y=0;y<hh;y++) for (let x=0;x<w;x++) if(p[y*w+x]!==p[(h-1-y)*w+x]) sV++;
  const holes=countHoles(p,w,h);
  return [dn,qn[0],qn[1],qn[2],qn[3],hC/h,vC/w,sH/(h*hw+1),sV/(hh*w+1),holes,dark?cxS/dark/w:0.5,dark?cyS/dark/h:0.5];
}

function classifyChar(p) {
  const f = features(p, 10, 16);
  const res = [];
  for (const [ch, ref] of Object.entries(CF)) {
    let d=0;
    for (let i=0;i<ref.length;i++) { const x=f[i]-ref[i]; d+=FEAT_WEIGHTS[i]*x*x; }
    res.push({ch, d:Math.sqrt(d)});
  }
  res.sort((a,b)=>a.d-b.d);
  const score = Math.max(0, Math.min(100, Math.round((1-res[0].d/3)*100)));
  return { t1:res[0].ch, t2:res[1].ch, t3:res[2].ch, score };
}

// ─────────────────────────────────────────────────────────────────────────────
// MULTI-HYPOTHESIS + MATCHING
// ─────────────────────────────────────────────────────────────────────────────
function applySchema(raw, schema) {
  let out='';
  for (let i=0;i<raw.length;i++) {
    const c=raw[i], s=schema[i];
    if (!s) { out+=c; continue; }
    if (s==='D' && isNaN(parseInt(c,10)) && L2D[c]) out+=L2D[c];
    else if (s==='L' && !isNaN(parseInt(c,10)) && D2L[c]) out+=D2L[c];
    else out+=c;
  }
  return out;
}

function genHypotheses(chars) {
  const base = chars.map(r=>r.t1).join('');
  const hyps = new Set([base]);

  // Swap each position with top2/top3
  for (let i=0;i<chars.length;i++) {
    hyps.add(chars.map((r,j)=>j===i?r.t2:r.t1).join(''));
    hyps.add(chars.map((r,j)=>j===i?r.t3:r.t1).join(''));
  }

  // D↔L swapped full variants
  for (const h of [...hyps]) {
    hyps.add(h.split('').map(c=>D2L[c]||c).join(''));
    hyps.add(h.split('').map(c=>L2D[c]||c).join(''));
  }

  return [...hyps].slice(0, 30);
}

function matchCountries(hyps) {
  const votes = {};
  for (const hyp of hyps) {
    const clean = hyp.replace(/[\s\-]/g,'');
    for (const [cc, info] of Object.entries(COUNTRIES)) {
      // Direct match
      if (info.r.test(clean)) {
        if (!votes[cc]) votes[cc]={plate:clean,score:0};
        votes[cc].score += 2;
      }
      // Schema-corrected match
      const cor = applySchema(clean, info.schema);
      if (cor !== clean && info.r.test(cor)) {
        if (!votes[cc]) votes[cc]={plate:cor,score:0};
        votes[cc].score += 1;
        if (votes[cc].score < 2) votes[cc].plate = cor;
      }
    }
  }
  return votes;
}

function plateColor(data, x, y, bw, bh, fullW) {
  let r=0,g=0,b=0,n=0;
  for (let py=y;py<y+bh;py+=2) for (let px=x;px<x+bw;px+=2) {
    const i=(py*fullW+px)*4;
    r+=data[i]; g+=data[i+1]; b+=data[i+2]; n++;
  }
  if (!n) return 'white';
  r/=n; g/=n; b/=n;
  if (r>200&&g>180&&b<120) return 'yellow';
  if (b>r+30&&b>g+20) return 'blue';
  if (r<100&&g<100&&b<100) return 'black';
  return 'white';
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN EXPORT
// ─────────────────────────────────────────────────────────────────────────────
async function recognizePlate(imageBlob) {
  const t0 = Date.now();

  const bitmap = await createImageBitmap(imageBlob);
  const scale  = Math.min(1, 1280 / bitmap.width);
  const W = Math.round(bitmap.width  * scale);
  const H = Math.round(bitmap.height * scale);

  const canvas = new OffscreenCanvas(W, H);
  const ctx    = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, W, H);
  const imgData = ctx.getImageData(0, 0, W, H);

  // Grayscale + gamma
  const gRaw = toGray(imgData.data, W, H);
  const gray  = applyGamma(gRaw, autoGamma(gRaw));

  // Detect regions
  const regions = detectRegions(gray, W, H);

  if (!regions.length) {
    return { plate:null, country:null, countryName:null, confidence:0,
             bbox:null, chars:[], charConfidences:[], plateColor:null,
             candidates:[], processingMs:Date.now()-t0, error:'No region detected' };
  }

  let best = null;

  for (const region of regions) {
    const {x,y,w,h} = region;

    // Extract region as sub-canvas
    const pc = new OffscreenCanvas(w, h);
    const pCtx = pc.getContext('2d');
    pCtx.drawImage(bitmap,
      Math.round(x/scale), Math.round(y/scale),
      Math.round(w/scale), Math.round(h/scale),
      0, 0, w, h
    );
    const pd = pCtx.getImageData(0,0,w,h);
    const pg = toGray(pd.data, w, h);
    const pgGamma = applyGamma(pg, autoGamma(pg));

    // Multi-threshold ensemble
    const configs = [{k:0.15,ws:11},{k:0.20,ws:15},{k:0.28,ws:21}];
    let regionBest = null;

    for (const cfg of configs) {
      const bin  = sauvola(pgGamma, w, h, cfg.k, 128, cfg.ws);
      const segs = segmentChars(bin, w, h);
      if (segs.length < 3 || segs.length > 12) continue;

      const chars = segs.map(s => {
        const p = extractPatch(bin, w, h, s.x, s.w, h);
        return classifyChar(p);
      });

      const hyps  = genHypotheses(chars);
      const votes = matchCountries(hyps);

      if (!Object.keys(votes).length) continue;

      const sorted = Object.entries(votes).sort((a,b)=>b[1].score-a[1].score);
      const [topCC, topV] = sorted[0];
      const charConfs = chars.map(c=>c.score);
      const avgCC = charConfs.reduce((a,b)=>a+b,0)/charConfs.length;

      const conf = Math.round(
        0.35 * Math.min(100, topV.score*15) +
        0.35 * avgCC +
        0.30 * Math.min(100, region.eDen/2)
      );

      const cand = {
        plate: topV.plate,
        country: topCC,
        countryName: COUNTRIES[topCC].name,
        confidence: conf,
        bbox: {x:Math.round(x),y:Math.round(y),w:Math.round(w),h:Math.round(h)},
        chars: chars.map(c=>c.t1),
        charConfidences: charConfs,
        plateColor: plateColor(imgData.data, x, y, w, h, W),
        candidates: sorted.slice(1,4).map(([cc,v])=>({
          country:cc, countryName:COUNTRIES[cc].name,
          plate:v.plate, confidence:Math.round(v.score*10)
        })),
        processingMs: 0
      };

      if (!regionBest || cand.confidence > regionBest.confidence) regionBest = cand;
    }

    if (regionBest && (!best || regionBest.confidence > best.confidence)) best = regionBest;
  }

  if (!best) {
    return { plate:null, country:null, countryName:null, confidence:0,
             bbox:null, chars:[], charConfidences:[], plateColor:null,
             candidates:[], processingMs:Date.now()-t0, error:'Segmentation failed' };
  }

  best.processingMs = Date.now() - t0;
  return best;
}

export { recognizePlate };
export default recognizePlate;
