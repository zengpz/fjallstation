# The task image: OpenModelica with the Modelica libraries of the tasks, and the two commands
# `fjallstation-task start` and `fjallstation-task submit` (see fjallstation_task.py).
# The agent of a user works in a container of this image. At the end the container itself tries the model of
# the agent and makes the result file that goes to the server. The image holds no task and no reference
# outputs. A container needs no network.
#
#   fjallstation image          (or: docker build -t fjallstation-env:1 <this folder>)
FROM openmodelica/openmodelica:v1.27.0-minimal

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-numpy python3-scipy \
 && rm -rf /var/lib/apt/lists/*

# Libraries with fixed versions, from the OpenModelica package manager:
# Buildings 13.0.0 with Modelica Standard Library 4.1.0, ModelicaServices and Complex.
RUN mkdir -p /opt/om \
 && printf 'updatePackageIndex(); getErrorString();\ninstallPackage(Buildings, "13.0.0", exactMatch=true); getErrorString();\n' > /opt/om/install.mos \
 && HOME=/opt/om omc /opt/om/install.mos \
 && test -f "/opt/om/.openmodelica/libraries/Buildings 13.0.0/package.mo" \
 && test -f "/opt/om/.openmodelica/libraries/Modelica 4.1.0+maint.om/package.mo" \
 && mv /opt/om/.openmodelica/libraries /opt/omlibrary \
 && rm -rf /opt/om \
 && chmod -R a+rX /opt/omlibrary \
 && mkdir -p /work && chmod 1777 /work

# The two commands. They belong to root; the agent works as a normal user and cannot change them.
COPY fjallstation_task.py /usr/local/bin/fjallstation-task
RUN chmod 755 /usr/local/bin/fjallstation-task && mkdir -p /etc/fjallstation && chmod 755 /etc/fjallstation

# FJALLSTATION_IMAGE is the version that a result file names. Change it together with the tag of the image.
ENV OPENMODELICALIBRARY=/opt/omlibrary \
    HOME=/tmp \
    FJALLSTATION_IMAGE=fjallstation-env:1
WORKDIR /work
CMD ["sleep", "infinity"]
