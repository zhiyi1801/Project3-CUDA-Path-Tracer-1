CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**


![](gallery/glassbunny2.png)
![](gallery/camera.png)

TODO:
- [ ] MIS
    - [x] BSDF sample(Lambertian, Dielectric, Microfacet, Metallic)
    - [ ] Light sample(light of different shape: sphere, cube, plane, triangles)
    - [ ] Integrate these two sample strategies
    - [ ] MIS based on luminance of light
    - [ ] Prove the unbiasedness of this method
- [ ] DOF
- [ ] Denoising
    - [ ] OpenImage Denoiser built [OpenImage](https://www.openimagedenoise.org/)
        - CPU only for now
    - [ ] Integrate it into project