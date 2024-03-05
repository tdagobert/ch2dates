FROM registry.ipol.im/ipol:v1-py3.9

RUN pip install --no-cache-dir numpy scipy numba iio matplotlib

#RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /workdir/bin
RUN mkdir -p /workdir/exec

ENV bin /workdir/bin/
WORKDIR /workdir/bin
COPY . .

ENV HOME /workdir/exec
WORKDIR /workdir/exec
