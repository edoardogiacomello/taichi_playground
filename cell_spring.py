import taichi as ti
ti.init(arch=ti.gpu)

particle_struct = {
    "pos": ti.math.vec2,
    "vel": ti.math.vec2,
    "acc": ti.math.vec2,
    "mass": float,
    "cell_id": int,
    "is_nucleus": int,
    "is_attached": int,
}

@ti.data_oriented
class Simulation():
    def __init__(self, H=1024, W=1024, dt=1e-4, headless=False):
        self.H = H
        self.W = W
        # Using old GUI to support Vulkan 1.4 on some machines
        # self.window = ti.ui.Window(name="MycroVerse", res=(H, W), show_window=not headless)
        # self.canvas = self.window.get_canvas()
        self.gui = ti.GUI(name="MycroVerse", res=max(H, W))
        self.time = ti.field(dtype=float, shape=())
        self.dt = dt

        # Create Nodes sparse structure
        self.S_particles = ti.root.dynamic(ti.i, 10000, chunk_size=32)
        self.particles = ti.Struct.field(particle_struct)
        self.S_particles.place(self.particles)


        self.max_particles = ti.field(dtype=int, shape=(1,))
        self.max_particles[0] = 10000
        # Positions of each particle
        self.pos = ti.Vector.field(2, dtype=float, shape=self.max_particles[0])
        # Array to store the object id of each particle
        self.part_to_obj = ti.field(dtype=int, shape=self.max_particles[0])
        self.part_to_obj.fill(-1)
        
        # UI Vars
        self.selected_cell = -1
    
    @ti.kernel
    def update_particles(self):
        pass

    def update(self):
        self.update_particles()

    @ti.kernel
    def reset(self):
        self.time[None] += self.dt
        pass


    def draw(self):
        # Draw particles
        to_draw = self.particles.pos.to_numpy()
        self.gui.circles(to_draw, radius=1, color=0xFF0000)
        self.gui.show()
    
    @ti.kernel
    def add_particle(self, x: float, y: float, cell_id: int):

        print("Adding Particle")
        first_free_idx = self.particles.length()
        self.particles[first_free_idx].is_nucleus = 1
        self.particles[first_free_idx].pos[0] = x
        self.particles[first_free_idx].pos[1] = y
        print(self.particles[first_free_idx].pos)
        print(self.particles.length())

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
            if self.gui.is_pressed("a"):
                x, y = self.gui.get_cursor_pos()                
                self.add_particle(x, y, self.selected_cell)
            
    def run(self):
        self.reset()
        while self.gui.running:
            self.parse_input()
            self.update()
            self.draw()

sim = Simulation()
sim.run()