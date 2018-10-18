FROM ubuntu:18.04
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip nano net-tools

RUN mkdir -p /opt/whalebone/ /etc/whalebone/logs /etc/whalebone/compose /etc/whalebone/cli/

RUN pip3 install "docker==3.0.1" psutil "websockets==4.0.1" pyaml netifaces dnspython cryptography requests

WORKDIR /opt/whalebone/

COPY . .
#RUN chown whalebone /opt/whalebone/ -R && chgrp whalebone /opt/whalebone/ -R && chmod g+s /opt/whalebone/ -R

#USER whalebone
CMD ["/opt/whalebone/lr_agent.sh"]
