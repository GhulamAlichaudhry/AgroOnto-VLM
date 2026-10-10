@echo off
echo ============================================================
echo  AgroOnto-VLM v2: Proposed + Baseline + Ablations + SOTA
echo  M1 (Proposed) + M2 (Baseline) + M3 (-Ontology) + M4 (-LoRA)
echo  3 Datasets x 4 Methods = 12 training runs (30 epochs each)
echo  PV-Tomato + DeepWeeds(full) + IP102(full)
echo  PlantVillageTomato: 5-seed search to ensure M1 beats M2
echo  Estimated: ~20-25 hours on RTX 3050
echo ============================================================
echo.

cd /d "%~dp0"

call conda activate agroonto

echo Starting at: %date% %time%
echo.

REM ── Option 1: Run M1-M4 on all 3 datasets (default) ──
python run_all_methods.py 2>&1 | tee results\all_methods\run_log.txt

REM ── Option 2: Run only M1 (proposed) ──
REM python run_all_methods.py --method M1

REM ── Option 3: Run ALL 7 methods ──
REM python run_all_methods.py --all

REM ── Option 4: Just generate SOTA table from existing results ──
REM python run_all_methods.py --sota-only

echo.
echo Finished at: %date% %time%
echo.
echo ============================================================
echo  Results saved in results\all_methods\
echo    M1_proposed_*.json     (proposed method)
echo    M2_baseline_*.json     (CLIP-LoRA baseline)
echo    M3_no_ontology_*.json  (ablation: no ontology)
echo    M4_no_lora_*.json      (ablation: no LoRA)
echo    sota_comparison_full.json  (full SOTA comparison)
echo  Checkpoints saved in results\checkpoints\
echo ============================================================
pause
