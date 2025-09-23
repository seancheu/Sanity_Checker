"""
python infer_tcn_bilstm.py
  --slices_h5 slices_out/slices.h5
  --meta_json slices_out/meta.json
  --ckpt /path/to/your/best_model.pt
  --classes "OOK,BPSK,QPSK,8PSK,16QAM,64QAM,256QAM,AM-SSB-WC,AM-SSB-SC,FM,GMSK,OQPSK"
  --extra_ch "amp,phase,dfreq,cosphi,sinphi,d_amp,d2phase,cum40,cum41,cum42"
  --T_crop 16384 --blocks 10 --kernel 7 --rnn_hidden 384
  --out_dir step6_out

This writes:
step6_out/predictions.csv (clip_index, pred, confidence)
step6_out/detections.csv (joined with freq/time/bw from meta)

python summarize_band.py
  --detections_csv step6_out/detections.csv
  --features_csv step4_out/features.csv
  --out_dir step6_out



"""