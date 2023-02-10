train:
	python ./main.py --data_path ntu_skeleton --batch_size 128 --epochs 1000

load_model:
	python ./main.py --data_path ntu_skeleton --batch_size 512 --epochs 500 --pretrained_encoder saved_model/encoder2.pt --pretrained_decoder saved_model/decoder2.pt

encode:
	python encode.py --data_path ntu_skeleton --output_path output/encode_data.csv --pretrained_encoder saved_model/encoder3.pt
