TAG="synister"

TMP_FILE:=$(shell mktemp).img

.PHONY: default push
default:
	cp -rp ../synister .
	cp -p ../setup.py .
	cp -p ../requirements.txt .
	sudo singularity build $(TMP_FILE) Singularity
	cp $(TMP_FILE) $(TAG).img
	sudo rm $(TMP_FILE)
