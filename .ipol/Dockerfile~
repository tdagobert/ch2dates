FROM registry.ipol.im/ipol:v1-py3.9

RUN pip install --no-cache-dir seaborn scikit-learn==1.2.1

RUN mkdir -p /workdir/bin
RUN mkdir -p /workdir/exec

ENV bin /workdir/bin/
WORKDIR /workdir/bin
COPY . .
#RUN cd src_random_mask  &&  make -j  &&  mv random_mask /workdir/bin/ 
#RUN cmake  -D IPOL_DEMO=1 .  &&  make -j 

#COPY .ipol/run.sh /workdir/bin

#RUN python3 prefetch_data.py mnist_784 /workdir/bin/ && rm -fr /scikit_learn_data/openml/openml.org/data/v1/download/
#RUN python3 prefetch_data.py CIFAR_10_small /workdir/bin/ && rm -fr /scikit_learn_data/openml/openml.org/data/v1/download/

ENV HOME /workdir/exec
WORKDIR /workdir/exec
