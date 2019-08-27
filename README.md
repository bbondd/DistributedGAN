# DistributedGAN
DistributedGAN

Distributed GAN Presentation.pptx is presentation file

Generating MNIST dataset

Multiple generator and discriminator in a distributed environment

Exchanging generator and discriminator for learning speed and quality improvement

Hyperparameter search through genetic algorithms

Using Tensorflow, Ray, Python

Currently incomplete


Architecture of model

![architecture](./results/architecture.png)

Example of 3 discriminator and 3 generator with 3 federated learning. It can be distributed to 3 * 3 * 3 + 1(main server) = 28 computers.

You can change the number of discriminator, generator, and federated learning.(ex. 4d, 2g, 3fl => 4 * 2 * 3 + 1 = 25 computers)

These are the results of distributed learning
![architecture](./results/1g1d.png)
![architecture](./results/1g2d.png)
![architecture](./results/2g1d.png)
![architecture](./results/2g2d.png)
![architecture](./results/1g1d2fl.png)

No significant quality improvement was seen.

Also, the speed of learning is slower because of the bottleneck caused by ray's distributed system through the Internet.

In gtx1060, 3 Generator, 2 Discriminator, 1 Federated Learning, it takes 14sec on 1 computer and 1min 17sec on 6 computers in 1 iteration.
