import taichi as ti
ti.init(arch=ti.vulkan)



@ti.data_oriented
class Simulation():
    def __init__(self, H=1024, W=1024, dt=1e-4, headless=False):
        self.H = H
        self.W = W
        self.window = ti.ui.Window(name="MycroVerse", res=(H, W), show_window=not headless)
        self.canvas = self.window.get_canvas()
        self.time = ti.field(dtype=float, shape=())
        self.dt = dt
        self.max_particles = ti.field(dtype=int, shape=(1,))
        self.max_particles[0] = 10000
        # Positions of each particle
        self.pos = ti.Vector.field(2, dtype=float, shape=self.max_particles[0])
        # Array to store the object id of each particle
        self.part_to_obj = ti.field(dtype=int, shape=self.max_particles[0])
        self.part_to_obj.fill(-1)
        
        # UI Vars
        self.selected_cell = -1
    
    def update(self):
        pass

    @ti.kernel
    def reset(self):
        self.time[None] += self.dt
        pass


    def draw(self):
        self.canvas.set_background_color((1, 1, 1))
        # Draw particles
        self.canvas.circles(self.pos, radius=.01, color=(0, 0, 0))
        self.window.show()
    
    @ti.kernel
    def add_particle(self, x: float, y: float, cell_id: int):
        # TODO: DOES NOT WORK
        n = self.max_particles[0]
        drawn = False
        for i in range(n):
            if self.part_to_obj[i] == -1 and not drawn:
                drawn = True
                self.part_to_obj[i] = cell_id
                self.pos[i] = [x, y]
                print(f"Adding particle at {x}, {y} with cell id {cell_id}")

    def parse_input(self):
        events = self.window.get_events()
        for e in events:
            if e.key == ti.ui.ESCAPE:
                self.window.destroy()
            if e.key == 'r':
                self.reset()
            # If number key is pressed, change selected cell
            if self.window.is_pressed(ti.ui.UP):
                self.selected_cell = max(0, self.selected_cell + 1)
                print(f"Selected cell: {self.selected_cell}")
            if self.window.is_pressed(ti.ui.DOWN):
                self.selected_cell = max(0, self.selected_cell - 1)
                print(f"Selected cell: {self.selected_cell}")
            if self.window.is_pressed("a"):
                print("Adding particle")
                x, y = self.window.get_cursor_pos()
                
                self.add_particle(x, y, self.selected_cell)
            
    def run(self):
        self.reset()
        while self.window.running:
            self.parse_input()
            self.update()
            self.draw()

sim = Simulation()
sim.run()