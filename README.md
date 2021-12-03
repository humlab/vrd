The Video Reuse Detector

# Download Docker
The VRD is designed to be run and installed through Docker – a third-party application that enables the use of containerized applications. To begin with, you must therefore download and install the Docker software.
[Click here](https://docs.docker.com/get-docker/) to find the Docker installation packages and instructions for your operating system.

# Install git
To download the necessary VRD source code, you need to install git, a version control system. Instructions for your operating system can be found [here](https://git-scm.com/downloads).

Install the VRD Docker container
After the prerequisites have been installed, create a subdirectory where you want your VRD install to be located, open a terminal (or equivalent) and navigate to newly created subdirectory. Grab the latest version by typing:

´git clone https://github.com/humlab/vrd.git´

When the command has finished, type

´docker-compose up´ to start the VRD system. The first time this command is run, it may take some time.


