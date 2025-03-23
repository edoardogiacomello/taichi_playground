import taichi as ti
import math
import numpy as np


ti.init(arch=ti.gpu)

# TODO: Make also self.grid sparse

@ti.data_oriented
class Simulation():
    def __init__(self, H=512, W=512, dt=1e-4, headless=False):

        # Physics properties
        self.damping = 0. # Damp velocities to simulate viscosity

        self.H = H
        self.W = W
        # Using old GUI to support Vulkan 1.4 on some machines
        # self.window = ti.ui.Window(name="MycroVerse", res=(H, W), show_window=not headless)
        # self.canvas = self.window.get_canvas()
        self.gui = ti.GUI(name="MycroVerse", res=max(H, W))
        self.time = ti.field(dtype=float, shape=())
        self.dt = dt


        # self.T_cell = ti.types.struct(cell_id=int, radius=int, particles_idx=int)
        # self.S_cells = ti.root.dynamic(ti.i, 1, chunk_size=10)
        # self.cells = self.T_cell.field()
        # self.S_cells.place(self.cells)
        
        # IDs of the cells in the simulation
        self.cells = []

        # Create particles as a dynamic struct
        self.T_particle = ti.types.struct(pos=ti.math.vec2,
                                          vel=ti.math.vec2,
                                          acc=ti.math.vec2,
                                          mass=float,
                                          cell_id=int,
                                          is_nucleus=int,
                                          is_membrane=int,
                                          is_attached=int)
        self.S_particles = ti.root.dynamic(ti.i, 1000, chunk_size=32)
        self.particles = self.T_particle.field()
        self.S_particles.place(self.particles)


        # MPM Grids (Physics)
        self.grid_v = ti.Vector.field(2, dtype=float, shape=(self.H, self.W))  # Velocities
        self.grid_m = ti.field(dtype=float, shape=(self.H, self.W))  # Mass


        # What is displayed on screen
        self.render_grid = ti.field(dtype=float, shape=(H, W))
    
    @ti.kernel
    def p2g(self):
        """
            Particle to Grid for the M2M Solver
        """
        for p in range(self.particles.length()):
            part = self.particles[p]
            base = ti.cast(part.pos * [self.W, self.H], int)  # Find grid cell
            fx = part.pos * [self.W, self.H] - base  # Fractional position

            w = [0.5 * (1.5 - fx) ** 2, 
                0.75 - (fx - 1.0) ** 2, 
                0.5 * (fx - 0.5) ** 2]

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    weight = w[i].x * w[j].y
                    self.grid_v[base + ti.Vector([i, j])] += weight * part.mass * part.vel
                    self.grid_m[base + ti.Vector([i, j])] += weight * part.mass

    @ti.kernel
    def grid_step(self):
        for i, j in self.grid_v:
            if self.grid_m[i, j] > 0:
                self.grid_v[i, j] /= self.grid_m[i, j]  # Normalize velocity
                self.grid_v[i, j] *= 0.98  # Apply damping

    @ti.kernel
    def g2p(self):
        for p in range(self.particles.length()):
            part = self.particles[p]
            base = ti.cast(part.pos * [self.W, self.H], int)
            fx = part.pos * [self.W, self.H] - base

            w = [0.5 * (1.5 - fx) ** 2, 
                0.75 - (fx - 1.0) ** 2, 
                0.5 * (fx - 0.5) ** 2]

            new_v = ti.Vector([0.0, 0.0])
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    weight = w[i].x * w[j].y
                    new_v += weight * self.grid_v[base + ti.Vector([i, j])]

            part.vel = new_v
            # if part.is_attached:
            #     part.vel = ti.Vector([0.0, 0.0])
            part.pos += part.vel * self.dt  # Move particle


    @ti.kernel
    def apply_jitter(self):
        for p in range(self.particles.length()):
            self.particles[p].pos += [(ti.random(float)-.5)/100., (ti.random(float)-.5)/100.]

    def update(self):
        #self.apply_jitter()

        self.p2g()      # Particle to grid
        self.grid_step()  # Apply grid-based operations (damping, forces)
        self.g2p()      # Grid to particle
        

    def spawn_cell(self, x: float = .5, y: float = .5, radius:float=.3):
        cell_id = len(self.cells) + 1
        self.add_cell(xc=x, yc=y, radius=radius, cell_id=cell_id)
        self.cells.append(cell_id)
        print(f"Added cell {cell_id}")
    
    @ti.kernel
    def add_cell(self, xc: float, yc: float, radius: float, cell_id: int):
        
        T = 4
        R = 4
        
        # Add nucleus
        nucleus = self.T_particle(pos=[xc, yc], 
                                    mass=1, 
                                    cell_id=cell_id, 
                                    is_nucleus=1, 
                                    is_membrane=0)
        self.particles.append(nucleus)

        for r in range(R):
            inner_radius = radius * (1 - r / R)
            for t in range(T):
                theta = 2 * math.pi * t / T
                x = xc + inner_radius * ti.cos(theta)
                y = yc + inner_radius * ti.sin(theta)
                is_membrane = 1 if r == 0 else 0  # Outer circle gets is_membrane = 1
                new_particle = self.T_particle(pos=[x, y], 
                                                mass=1, 
                                                cell_id=cell_id, 
                                                is_nucleus=0, 
                                                is_membrane=is_membrane)
                self.particles.append(new_particle)
        

    def reset(self):
        print(f"Simulation Reset")
        self.time[None] += self.dt        
        self.spawn_cell(x=.7, y=.7, radius=.1)
        self.spawn_cell(x=.3, y=.3, radius=.1)
        pass
    
    @ti.kernel
    def compute_density_map(self):
        """
            Render each pixel as the sum of the particles that are in that position
        """
        self.render_grid.fill(0.0)
        for p in range(self.particles.length()):
            part = self.particles[p]
            base = ti.cast(part.pos * [self.W, self.H], int)
            self.render_grid[base.x, base.y] += 1.0  # Increment density


    def draw(self):
        self.gui.clear()
        # Draw particles
        self.compute_density_map()
        self.gui.set_image(self.render_grid)
        self.gui.show()
    

    def parse_input(self):
        events = self.gui.get_events()
        for e in events:
            if e.key == ti.ui.ESCAPE:
                self.gui.destroy()
            if e.key == 'r':
                self.reset()
            if e.key == ti.ui.LMB:  # Left Mouse Button Click
                self.closest_idx = ti.field(dtype=ti.i32, shape=(10000))  # Store closest index
                mx, my = self.gui.get_cursor_pos()  # Get mouse position (0,1)
                closest_particle = self.find_closest_particle(mx, my)  # Read from field
                
                if closest_particle != -1:
                    print(f"Clicked particle {closest_particle}")
                    self.apply_force(closest_particle, mx, my)

    @ti.kernel
    def find_closest_particle(self, mx: float, my: float) -> int:
        min_dist = ti.math.inf  # Large number
        click_pos = ti.Vector([mx, my])
        min_idx = -1

        for p in range(self.particles.length()): 
            dist = (self.particles[p].pos - click_pos).norm()
            ti.atomic_min(min_dist, dist)
        
        for p in range(self.particles.length()):
            dist = (self.particles[p].pos - click_pos).norm()
            if dist == min_dist:
                min_idx = p
        
        return min_idx
    
    @ti.kernel
    def apply_force(self, p_id:int, mx: float, my: float):
        
        # Apply velocity towards the click direction
        from_xy = ti.Vector([mx, my]) 
        direction = (from_xy - self.particles[p_id].pos).normalized()
        self.particles[p_id].vel += direction * 0.5  # Adjust strength

    def run(self):
        self.reset()
        while self.gui.running:
            self.parse_input()
            self.update()
            self.draw()

sim = Simulation()
sim.run()