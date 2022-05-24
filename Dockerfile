FROM huggingface/transformers-pytorch-gpu
EXPOSE 5000:5000
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install vim wget curl man git pip
RUN pip install --upgrade pip

# install server requirments
RUN pip install flask

# zsh
RUN apt-get -y install zsh

RUN chsh -s $(which zsh)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

# KoELECTRA requirments
RUN pip install seqeval

# COPY init.sh /root/init.sh
# CMD bash /root/init.sh