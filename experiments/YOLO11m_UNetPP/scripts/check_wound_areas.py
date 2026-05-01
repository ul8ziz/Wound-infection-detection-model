#!/usr/bin/env python3
"""Quick check of wound area measurements, marker detection, and infection status."""
import json
import statistics
from pathlib import Path

results_path = Path("experiments/YOLO11m_UNetPP/results/combined")
candidates = sorted(results_path.glob("*/wound_areas.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not candidates:
    candidates = [results_path / "wound_areas.json"]

wound_areas_path = candidates[0]
print(f"Reading: {wound_areas_path}\n")

with open(wound_areas_path, "r") as f:
    data = json.load(f)

detected = [x for x in data if x.get("marker_detected", False)]
not_detected = [x for x in data if not x.get("marker_detected", False)]

print(f"Total images: {len(data)}")
print(f"Marker detected: {len(detected)} ({100*len(detected)/max(len(data),1):.1f}%)")
print(f"Marker NOT detected: {len(not_detected)} ({100*len(not_detected)/max(len(data),1):.1f}%)")
print()

if detected:
    ppcms = [x["pixels_per_cm"] for x in detected if x.get("pixels_per_cm")]
    areas = [x["area_cm2"] for x in detected if x.get("area_cm2") is not None]
    print("=== With marker (dynamic calibration) ===")
    if ppcms:
        print(f"  pixels_per_cm: min={min(ppcms):.2f}, max={max(ppcms):.2f}, "
              f"mean={statistics.mean(ppcms):.2f}, std={statistics.stdev(ppcms):.2f}")
    if areas:
        print(f"  area_cm2: min={min(areas):.2f}, max={max(areas):.2f}, "
              f"mean={statistics.mean(areas):.2f}, median={statistics.median(areas):.2f}")
    print()

if not_detected:
    print("=== Without marker (cm2 unavailable) ===")
    for x in not_detected:
        area_px = x.get("area_px", "N/A")
        print(f"  {x['image']:50s} area_px={area_px}")
    print()

# Infection stats
infected = [x for x in data if x.get("infection") == "infected"]
not_infected = [x for x in data if x.get("infection") == "not_infected"]
unknown = [x for x in data if x.get("infection") in (None, "unknown")]

print("=== Infection classification ===")
print(f"  Infected:     {len(infected)}")
print(f"  Not infected: {len(not_infected)}")
print(f"  Unknown:      {len(unknown)}")

if infected:
    probs = [x.get("infection_prob", -1) for x in infected if x.get("infection_prob", -1) >= 0]
    if probs:
        print(f"  Infected confidence: min={min(probs):.3f}, max={max(probs):.3f}, mean={statistics.mean(probs):.3f}")
print()

print("=== Sample measurements ===")
for x in data[:15]:
    marker_str = "MEASURED" if x.get("marker_detected") else "NO_SCALE"
    quality = x.get("measurement_quality", marker_str.lower())
    area_str = f"{x['area_cm2']:8.2f} cm2" if x.get("area_cm2") is not None else f"{x.get('area_px', 0):>8} px "
    inf_str = x.get("infection", "unknown")
    ppcm_str = f"{x['pixels_per_cm']:7.2f}" if x.get("pixels_per_cm") else "   N/A "
    print(f"  {x['image']:50s} {area_str}  ppcm={ppcm_str}  [{quality:11s}]  {inf_str}")
