#!/usr/bin/env node
/* Generates the app icons (PNG + SVG) with no dependencies: a small anti-aliased rasterizer
 * and a PNG encoder built on Node's zlib. Run: node tools/make-icons.js */
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const OUT = path.join(__dirname, '..', 'icons');
// A gilded medallion: gold field, a fresco-sky disc lit from above, the concept graph in ivory.
const G1 = [242, 210, 126], G2 = [138, 97, 22];                         // gold, light to deep
const SKY = [[255, 244, 214], [169, 203, 233], [58, 102, 159]];       // light at the centre, sky, lapis edge
const DISC = 0.4, IVORY = [255, 250, 240], RIM = [184, 137, 43];
const NODES = [[0.5, 0.29, 0.075], [0.29, 0.5, 0.066], [0.71, 0.5, 0.066], [0.5, 0.71, 0.075]];
const EDGES = [[0, 1], [0, 2], [1, 3], [2, 3], [1, 2]];
const LINE = 0.034;

function segDist(px, py, ax, ay, bx, by) {
  const vx = bx - ax, vy = by - ay; const t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)));
  return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
}
const mix = (a, b, t) => a.map((c, i) => c + (b[i] - c) * t);
function shade(u, v) {
  let rgb = mix(G1, G2, (u + v) / 2);                                  // the gold field
  const d = Math.hypot(u - 0.5, v - 0.5);
  if (d < DISC) {                                                      // the sky disc, brightest a little above the centre
    const t = Math.min(1, Math.hypot(u - 0.5, v - 0.42) / (DISC * 1.1));
    rgb = t < 0.42 ? mix(SKY[0], SKY[1], t / 0.42) : mix(SKY[1], SKY[2], (t - 0.42) / 0.58);
    for (const [a, b] of EDGES) if (segDist(u, v, NODES[a][0], NODES[a][1], NODES[b][0], NODES[b][1]) < LINE / 2) rgb = mix(rgb, IVORY, 0.9);
    for (const [x, y, r] of NODES) { const e = Math.hypot(u - x, v - y); if (e < r) rgb = IVORY; else if (e < r + 0.012) rgb = RIM; }
  } else if (d < DISC + 0.014) rgb = mix(G2, G1, 0.25);                // a darker rim around the disc
  return rgb;
}
function render(n) {
  const SS = 4; const px = Buffer.alloc(n * n * 4);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let r = 0, g = 0, b = 0;
    for (let sy = 0; sy < SS; sy++) for (let sx = 0; sx < SS; sx++) { const c = shade((x + (sx + 0.5) / SS) / n, (y + (sy + 0.5) / SS) / n); r += c[0]; g += c[1]; b += c[2]; }
    const k = (y * n + x) * 4; const d = SS * SS;
    px[k] = Math.round(r / d); px[k + 1] = Math.round(g / d); px[k + 2] = Math.round(b / d); px[k + 3] = 255;
  }
  return px;
}
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(buf) { let c = 0xffffffff; for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function chunk(type, data) { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type, 'ascii'), data]); const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td)); return Buffer.concat([len, td, crc]); }
function png(n, rgba) {
  const raw = Buffer.alloc((n * 4 + 1) * n);
  for (let y = 0; y < n; y++) { raw[y * (n * 4 + 1)] = 0; rgba.copy(raw, y * (n * 4 + 1) + 1, y * n * 4, (y + 1) * n * 4); }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(n, 0); ihdr.writeUInt32BE(n, 4); ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
}
function svg() {
  const s = v => (v * 512).toFixed(1);
  const edges = EDGES.map(([a, b]) => `<line x1="${s(NODES[a][0])}" y1="${s(NODES[a][1])}" x2="${s(NODES[b][0])}" y2="${s(NODES[b][1])}"/>`).join('');
  const nodes = NODES.map(([x, y, r]) => `<circle cx="${s(x)}" cy="${s(y)}" r="${s(r)}"/>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="rgb(${G1})"/><stop offset="1" stop-color="rgb(${G2})"/></linearGradient><radialGradient id="s" cx="50%" cy="42%" r="55%"><stop offset="0" stop-color="rgb(${SKY[0]})"/><stop offset=".42" stop-color="rgb(${SKY[1]})"/><stop offset="1" stop-color="rgb(${SKY[2]})"/></radialGradient></defs><rect width="512" height="512" rx="112" fill="url(#g)"/><circle cx="256" cy="256" r="${s(DISC + 0.014)}" fill="rgb(${mix(G2, G1, 0.25).map(Math.round)})"/><circle cx="256" cy="256" r="${s(DISC)}" fill="url(#s)"/><g stroke="rgb(${IVORY})" stroke-opacity=".9" stroke-width="${s(LINE)}" stroke-linecap="round">${edges}</g><g fill="rgb(${IVORY})" stroke="rgb(${RIM})" stroke-width="${s(0.012)}">${nodes}</g></svg>\n`;
}
fs.mkdirSync(OUT, { recursive: true });
for (const n of [192, 512]) { fs.writeFileSync(path.join(OUT, 'icon-' + n + '.png'), png(n, render(n))); console.log('wrote icons/icon-' + n + '.png'); }
fs.writeFileSync(path.join(OUT, 'icon.svg'), svg()); console.log('wrote icons/icon.svg');
