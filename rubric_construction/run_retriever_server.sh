export JAVA_HOME=jdk-21.0.7
export CLASSPATH=.:${JAVA_HOME}/lib
export PATH=${CLASSPATH}:${JAVA_HOME}/bin:$PATH


python retriever_server.py
