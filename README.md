# DARE to D:REAM

A simulated Schwarzschild ray-marcher that renders a gravitationally lensed black hole, built to feed a head-tracked display and create an immersive simulation of black holes and other celestial bodies in dark viewing rooms.

**Status:** Hardware validated and working; simulation built.

- **Stage 1:** Test different ideas, validate the current one and learn the required physics ✓
- **Stage 2:** Design and build the hardware, decide on a plausible software architecture for an old processor, and test ✓
- **Stage 3:** Improve tracking and demo the entire build

After reading *Black Holes* by Brian Cox, I wanted to know what it would feel like to see a black hole in 3D in my room, with my own eyes. I was not completely satisfied with games like Space Engine or VR (although both are amazing). I wanted to see something 3D, physically accurate and close up.

The initial idea came from messing with a wine glass on my laptop, as seen below:

https://github.com/user-attachments/assets/c35a5ea6-3ab8-4285-ba8e-9eb87f224beb

After seeing the possibility, I decided to go for it. Here are the designs I evaluated before picking the quite novel technique that I designed from first principles.

## 1. Designs Considered

### Design 1: The overcomplicated glass bomb

<img width="1600" height="469" alt="Design 1" src="https://github.com/user-attachments/assets/d18cba2a-9615-49fe-a4b5-6e0c8db3c25b" />
The issues were as follows:

- **Viewing angles:** I could not get a good transition from about +10°, through dead-on, to −10°. The wine glass could only render accurately between 10° and 40° viewing angles.
- **The spinning fan:** A spinning fan was my cheapest option for illuminating an image of both the top and the bottom of an accretion disk. It had to be hand-built, adding another point of failure.
- **Power:** The biggest thorn in my side was getting enough current to a bus servo and high-refresh-rate LED light strips while spinning. I looked at using a 3.5 mm jack to build my own slip ring, and I searched for cheap slip rings, but there are none that are both high-current and high-RPM. I contemplated using a battery, but that felt like a cop-out and would add mass to the spinning elements.
- **Mirrors:** I decided mirrors might be an easier and more lightweight solution. I could selectively render different images on the fan depending on the viewing angle and reflect them back at the viewer with the mirror. I attempted to validate this concept on my laptop, and it did not work.
### Design 2: The less whimsical but clean approach

<img width="1600" height="883" alt="Design 2" src="https://github.com/user-attachments/assets/7205296b-a399-484a-854b-2f8d7b7bec20" />
The best analogy is viewing a bird on a branch through a window. Imagine the window moves with you, so that wherever you go to see the bird from a different angle, the window follows, and you can never look at the bird from any point of view that is not through the window. If you make the frame and the window pane invisible, you will be convinced you are simply looking at the bird.

Similarly, if you set up an OLED screen in a dark room and move it so that it always stays perpendicular to the plane of your face, then render the appropriate angle of the black hole on the screen, it will appear that you are seeing the black hole in real life.

**Issue:** You cannot avoid stereopsis, the fact that humans see two different angles of an object, one from each eye. That said, from about 2-3 m away this becomes unnoticeable.

<img width="1173" height="757" alt="prelim-designs-landscape" src="https://github.com/user-attachments/assets/0fa2b1f3-f936-4ff3-b886-2d17abedc421" />
Some drawings of trying to map the wiring, measure the wiring harness and fit all the electronics into the build.

## 2. The Code

### 2.1 The physics

Here are my original notes from the project:

<img align="right" src="https://github.com/user-attachments/assets/69c9e2df-a259-423f-8cbb-47bf04b54584" width="300" />

**Rejected approach 1: Newtonian.** Nudging the ray direction like this would work perfectly well, but it is not General Relativity, which is what I wanted to use:

$$\vec{d}_{\text{new}} = \vec{d} - \frac{\vec{r}\,\Delta s\,M}{|\vec{r}|^{3}}$$

**Rejected approach 2: Volumetric absorption.** Beer's law is also an option for volumetric effects, but I thought it would be too computationally heavy:

$$T = e^{-\alpha\,\Delta x}$$

That said, I later figured out a software trick that would make this feasible.

**What I chose:** Calculating the force that bends each ray, which is the behavior of a geodesic. It comes from the Schwarzschild metric (in units where the Schwarzschild radius is 1):

$$ds^{2} = \left(1 - \frac{1}{r}\right)dt^{2} - \left(1 - \frac{1}{r}\right)^{-1}dr^{2} - r^{2}\,d\theta^{2} - r^{2}\sin^{2}\theta\,d\phi^{2}$$

(In my original notes I guessed that $ds^{2}$ is the same as the proper time $\tau$. I was correct, but only for massive particles ($ds^{2} = d\tau^{2}$); for light, $ds^{2} = 0$. I also guessed that the last two terms, $-r^{2}\,d\theta^{2} - r^{2}\sin^{2}\theta\,d\phi^{2}$, were the "3D stuff". That was also correct: they are the angular part of the metric, which is the sphere around the hole.)

From this metric you can derive the orbit equation for a light ray, with $u = 1/r$:

$$\frac{d^{2}u}{d\phi^{2}} = -u\left(1 - \frac{3}{2}\,u\right)$$

and from that, the force law:

$$F = -\frac{3}{2}\,\frac{h^{2}}{r^{5}}$$

where $h$ is the angular momentum per unit mass and $r$ is the radius.

I used [Starless by Riccardo Antonelli](https://rantonels.github.io/starless/) as a guide and did the math myself, then got Claude to put all the code together.

### 2.2 The numerical method

*Designed by myself and given to Claude as pseudocode to implement.*

1. **Camera:** The camera sits at distance `cam_dist` at the chosen elevation, looking at the origin. There is one ray direction per pixel.
2. **Integration:** All rays are marched together as numpy arrays using semi-implicit (forward) Euler: `vel += F*dt; pos += vel*dt`.
3. **Horizon:** A ray with $r < 1$ is killed and stays black.
4. **Disk hit:** The disk is the $z = 0$ plane between `disk_in` and `disk_out`. A hit is detected when $z$ changes sign between two steps, which catches fast rays that jump over the plane.
5. **Sky:** Rays that never hit anything are left as background.
### 2.3 The hit-map architecture

The computational trick I alluded to above comes down to the seven-year-old Raspberry Pi not having enough compute to render in real time. I thought it might be easier to pre-render every frame of the black hole, store them all on the Raspberry Pi, and look up each frame based on where the person is viewing from. That would take an insane amount of storage space.

However, because the azimuthal axis is visually indistinguishable no matter where you are, you only need to render the frames from top to bottom of the black hole and reuse them for the entire viewing sphere.

## 3. Design Constraints

- **Compute budget:** Raspberry Pi 4B, with a latency below 20 ms to feel instant.
- **Display:** An OLED is essential for the illusion to look realistic. An LCD does not actually turn off black pixels, so in a dark room you can see a rectangular gray shadow behind the black hole.
- **Physical / optical:** The room needs to be completely pitch black, and the display must be viewed from at least 2 m away.
## 4. Known Limitations

- Forward Euler is first-order, so rays near the photon sphere accumulate error and the inner ring can look speckled at low resolution.
- The disk is flat and evenly lit: there is no Doppler beaming or gravitational redshift.
- There are no higher-order images of the disk beyond what the step count and resolution happen to capture.
## 5. Future Plans

- [ ] Optimize the face-tracking software.
- [ ] Consider replacing the Raspberry Pi 4B with a more capable machine-vision chip.
- [ ] Try to get hold of a secondhand OLED display.
- [ ] Project other objects, such as the ISS and planets.
- [ ] Create a visual representation of speed using the speed of light crossing the monitor, scaled to the size of the celestial body being viewed. For example, the top four rows of pixels could move at the scaled speed of light. For M87, the progression of those pixels lighting up would be extremely slow, almost unnoticeable, but for something like the Sun it would be very fast.
## 6. References and Credits

- Riccardo Antonelli, [*Ray-tracing a black hole* (Starless)](https://rantonels.github.io/starless/)
- Sean Carroll, *Lecture Notes on General Relativity* (arXiv:gr-qc/9712019)
- O. James, E. von Tunzelmann, P. Franklin and K. S. Thorne, "Gravitational Lensing by Spinning Black Holes in Astrophysics, and in the Movie Interstellar", *Classical and Quantum Gravity* 32, 065001 (2015)
