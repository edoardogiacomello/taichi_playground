import taichi as ti
import math

ti.init(arch=ti.gpu)

# TODO: Make also self.grid sparse

@ti.data_oriented
class Simulation():
    def __init__(self, H=512, W=512, dt=1e-4, headless=False):
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
        self.cells = []

        # Create Nodes sparse structure
        self.T_particle = ti.types.struct(pos=ti.math.vec2,
                                          vel=ti.math.vec2,
                                          acc=ti.math.vec2,
                                          mass=float,
                                          cell_id=int,
                                          is_nucleus=int,
                                          is_membrane=int,
                                          is_attached=int)
        self.S_particles = ti.root.dynamic(ti.i, 1, chunk_size=32)
        self.particles = self.T_particle.field()
        self.S_particles.place(self.particles)

        # UI Vars
        self.selected_cell = 0

        # What is displayed on screen
        self.render_grid = ti.field(dtype=float, shape=(H, W))
    
    @ti.kernel
    def update_particles(self):
        pass

    def update(self):
        self.update_particles()

    def spawn_cell(self, x: float = .5, y: float = .5, radius:float=.3):
        cell_id = len(self.cells) + 1
        self.add_cell(xc=x, yc=y, radius=radius, cell_id=cell_id)
        self.cells.append(cell_id)
        print(f"Added cell {cell_id}")
    
    @ti.kernel
    def add_cell(self, xc: float, yc: float, radius: float, cell_id: int):
        
        T = 16
        R = 8
        
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
        self.time[None] += self.dt        
        self.spawn_cell(x=.7, y=.7, radius=.1)
        self.spawn_cell(x=.3, y=.3, radius=.1)
        pass
    
    @ti.kernel
    def render(self):
        """
            Draw (sparse) particles to the grid.
        """

        # Sparse particles cannot (apparently) be drawn using gui primitives since .to_numpy() would include also deactivated chunks
        for p in range(self.particles.length()):
            part = self.particles[p]
            # TODO: Use quadratic interpolations?
            self.render_grid[int(part.pos[0]*self.W), int(part.pos[1]*self.H)] = part.cell_id


    def draw(self):
        # Draw particles
        self.render()
        self.gui.set_image(self.render_grid)
        self.gui.show()
    

    def parse_input(self):
        events = self.gui.get_events()
        for e in events:
            if e.key == ti.ui.ESCAPE:
                self.gui.destroy()
            if e.key == 'r':
                self.reset()
            # If number key is pressed, change selected cell
            if self.gui.is_pressed(ti.ui.UP):
                self.selected_cell = max(0, self.selected_cell + 1)
                print(f"Selected cell: {self.selected_cell}")
            if self.gui.is_pressed(ti.ui.DOWN):
                self.selected_cell = max(0, self.selected_cell - 1)
                print(f"Selected cell: {self.selected_cell}")
            # if self.gui.is_pressed("a"):
            #     x, y = self.gui.get_cursor_pos()                
            #     self.add_particle(x, y, self.selected_cell)
            
    def run(self):
        self.reset()
        while self.gui.running:
            self.parse_input()
            self.update()
            self.draw()

sim = Simulation()
sim.run()