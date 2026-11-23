/* Sample course: three short game-development lessons that use every kind of section, a quiz, a check, a chart, a
 * prediction and a practice problem. They show the lesson format in action; replace them with your own lessons
 * (docs/MAKING_A_COURSE.md) and list your files in tools/content-files.json. */
(function (global) {
  const MLA = (global.MLA = global.MLA || {});

  MLA.content.add([
  // =====================================================================================
  {
    id: 'game-loop',
    branch: 'start',
    title: 'The Game Loop and Delta Time',
    short: 'Game loop',
    level: 1,
    minutes: 25,
    prereqs: [],
    tags: ['game loop', 'frame', 'frame rate', 'fps', 'delta time', 'update', 'fixedupdate', 'fixed time step', 'unity'],
    summary: 'Every game repeats one short cycle many times a second: read the input, update the world, draw it. Scaling movement by the time each frame took keeps the game running at the same speed on fast and slow machines.',
    why: `A game is a program that never waits for you. Even when you press nothing, it keeps redrawing the world dozens of times a second, and everything that moves, from the player to a falling leaf, moves a little in each of those frames. Getting this loop right is the difference between a game that plays the same everywhere and one that runs twice as fast on a gaming PC as on a laptop.`,
    objectives: [
      'Describe the three steps of a frame: input, update, render',
      'Explain why movement must be scaled by delta time, and compute a position after one frame',
      'Tell apart Update (once per frame) and FixedUpdate (fixed time steps for physics) in Unity'
    ],
    sections: [
      { type: 'text', title: 'One frame at a time', md: `A film shows still pictures fast enough that they look like motion. A game does the same, except that it has to *compute* each picture first. It does so in a loop that runs as long as the game is open:

1. **Input**: read the keyboard, mouse, controller or touch screen.
2. **Update**: move characters, run the AI, check collisions, count down timers.
3. **Render**: draw the new state of the world on the screen.

One pass through the loop is a **frame**. The number of frames per second (**fps**) is the frame rate: 30 fps feels acceptable, 60 fps feels smooth, and fast monitors show 144 or more. The frame rate is not fixed: it drops when the scene gets busy and rises on faster hardware.

In Unity you never write this loop yourself. The engine runs it and calls your scripts at the right moments: \`Update()\` once per frame, before the frame is drawn.` },
      { type: 'image', title: 'The loop as a picture', src: 'assets/images/game-loop.svg', alt: 'A cycle of three boxes: Input (keys, mouse, pad), then Update (move, collide, AI), then Render (draw the frame), with an arrow from Render back to Input.', after: `One trip around the loop is one frame. At 60 frames per second the whole cycle has about 16.7 milliseconds, so slow work in Update delays the next picture.` },
      { type: 'text', title: 'Why delta time matters', md: `Suppose a character moves 0.1 units every frame. At 30 fps it covers 3 units per second, at 60 fps 6 units, at 144 fps 14.4 units: the faster the computer, the faster the game. The fix is to give speeds **per second** and multiply by the time the last frame took:

$$ x_new = x + v · Δt $$
:::where
x — the position before this frame, in units
v — the speed, in units per second
Δt — "delta time": the time the previous frame took, in seconds (about 0.0167 at 60 fps)
x_new — the position after this frame
:::

A slow frame has a large Δt and moves the character further; a fast frame has a small Δt and moves it less. Over one second the steps always add up to v units, whatever the frame rate. Unity gives you Δt as \`Time.deltaTime\`.` },
      { type: 'code', title: 'Moving with and without delta time', md: `This simulation runs one second of frames at three frame rates. One object moves a fixed 0.1 units per frame; the other moves 5 units per second times Δt.`, code: `def run(fps, seconds=1.0, step=0.1, speed=5.0):
    frames = round(fps * seconds)
    dt = 1 / fps
    per_frame = 0.0       # moves a fixed step every frame
    timed = 0.0           # moves speed * dt every frame
    for _ in range(frames):
        per_frame += step
        timed += speed * dt
    return per_frame, timed

for fps in [30, 60, 144]:
    a, b = run(fps)
    print(f"{fps:>3} fps: fixed step per frame {a:5.2f} units, speed x dt {b:5.2f} units")`, after: `The object scaled by Δt covers 5 units at every frame rate. The other one covers almost five times more distance on a 144 Hz monitor than at 30 fps.`, predict: { type: 'mcq', q: 'At 144 frames per second, how far does the object that moves a fixed 0.1 units per frame travel in one second?', options: ['14.40 units', '5.00 units', '1.44 units', '0.10 units'], answer: 0, explain: 'It moves 0.1 units in each of 144 frames: 144 · 0.1 = 14.4 units. Only the object scaled by delta time stays at 5 units per second.', expect: ['144 fps: fixed step per frame 14.40 units'] } },
      { type: 'chart', title: 'Distance covered in one second at three frame rates', md: `The same simulation as a chart.`, chart: { kind: 'bar', x: ['30 fps', '60 fps', '144 fps'], series: [{ name: 'Fixed 0.1 units per frame', y: [3, 6, 14.4] }, { name: '5 units per second × Δt', y: [5, 5, 5] }], xLabel: 'Frame rate', yLabel: 'Distance in one second (units)' }, after: `The bars for the fixed step grow with the frame rate; the delta-time bars stay level. Numbers from tools/charts/game_loop.py.` },
      { type: 'code', title: 'The same idea in Unity', lang: 'csharp', run: false, md: `A Unity script attached to an object. \`Update\` runs once per frame, and \`Time.deltaTime\` is the time the previous frame took.`, code: `using UnityEngine;

public class Mover : MonoBehaviour
{
    [SerializeField] float speed = 5f;    // units per second, editable in the Inspector

    void Update()
    {
        // Time.deltaTime: seconds since the previous frame
        transform.position += Vector3.right * speed * Time.deltaTime;
    }
}` },
      { type: 'check', question: { type: 'tf', q: 'Multiplying a speed in units per second by the frame time makes movement independent of the frame rate.', answer: true, explain: 'Over one second the frame times add up to one second, so the steps add up to the speed, at any frame rate.' } },
      { type: 'text', title: 'Fixed time steps for physics', md: `Physics simulations misbehave when Δt jumps around: a long frame can push one object deep into another before any collision check runs. So physics engines advance in **fixed** steps instead. Unity calls \`FixedUpdate()\` every 0.02 seconds of game time by default (50 times a second), however fast or slow the frames are.

Within one frame, Unity runs as many fixed steps as are due. It keeps the leftover time for the next frame:

| Frame time | Time owed before the frame | Fixed steps run | Left over |
| --- | --- | --- | --- |
| 16 ms | 0 ms | 0 | 16 ms |
| 16 ms | 16 ms | 1 | 12 ms |
| 50 ms | 12 ms | 3 | 2 ms |

Put movement driven by forces and rigidbodies in \`FixedUpdate\`, and input reading, animation and camera work in \`Update\`.` }
    ],
    when: `- Every game uses a loop like this; in an engine such as Unity it is built in, and your code runs inside it.
- Scale any change that happens over time (movement, rotation, timers, fading) by delta time.
- Use fixed time steps for physics and anything that must give the same result on every machine.`,
    pitfalls: `- Moving a fixed amount per frame: the game speed then depends on the computer.
- Multiplying by delta time twice, or not at all, after copying code between Update and FixedUpdate.
- Reading one-off input, such as a key press, in FixedUpdate: it can run zero times in a fast frame and miss the press.`,
    takeaways: [
      'A frame is one pass of input, update and render; frame rates vary from machine to machine and moment to moment.',
      'Give speeds per second and multiply by delta time, the duration of the last frame.',
      'Physics advances in fixed time steps (0.02 s by default in Unity), run as often as the elapsed time requires.'
    ],
    quiz: [
      { type: 'mcq', q: 'What are the three steps of one pass through the game loop?', options: ['Read input, update the world, render', 'Load, compile, run', 'Draw, wait, draw again', 'Save, load, quit'], answer: 0, explain: 'Each frame reads the input, updates the game state and draws the result. Then the loop starts again.' },
      { type: 'fill', q: 'A character moves at 6 units per second. The last frame took 0.05 s. How many units does it move this frame?', answer: ['0.3', '.3'], explain: 'Distance = speed · Δt = 6 · 0.05 = 0.3 units.' },
      { type: 'mcq', q: 'A game moves its player 2 units every frame. What happens on a computer that runs it at twice the frame rate?', options: ['The player moves twice as fast', 'Nothing changes', 'The player moves half as fast', 'The game refuses to run'], answer: 0, explain: 'Twice as many frames per second means twice as many 2-unit steps per second. Scaling by delta time removes this dependence.' },
      { type: 'tf', q: 'In Unity, FixedUpdate runs exactly once per frame.', answer: false, explain: 'FixedUpdate runs at fixed intervals of game time (0.02 s by default), so a frame may contain zero, one or several fixed steps.' },
      { type: 'mcq', q: 'Where should code that pushes a Rigidbody with forces usually go in Unity?', options: ['FixedUpdate', 'Update', 'Awake', 'OnGUI'], answer: 0, explain: 'Forces act during the physics steps, which run at the fixed rate of FixedUpdate.' }
    ],
    problems: [
      {
        id: 'fixed-time-steps',
        title: 'Fixed Time Steps',
        difficulty: 'easy',
        prompt: `A physics engine advances in fixed steps of \`fixed_ms\` milliseconds. After each frame it runs as many steps as the time owed allows, and keeps the leftover time for the next frame.

Write \`physics_steps(frame_ms, fixed_ms=20)\` that takes the durations of successive frames (whole milliseconds) and returns how many fixed steps run after each frame.

Example: \`physics_steps([16, 16, 16, 50])\` returns \`[0, 1, 1, 2]\`. After the first frame 16 ms are owed (no step); after the second, 32 ms (one step, 12 ms left); after the third, 28 ms (one step, 8 ms left); after the fourth, 58 ms (two steps, 18 ms left).`,
        starter: `def physics_steps(frame_ms, fixed_ms=20):
    steps = []
    for ms in frame_ms:
        steps.append(ms // fixed_ms)    # forgets the time left over from earlier frames
    return steps`,
        solution: `def physics_steps(frame_ms, fixed_ms=20):
    steps, owed = [], 0
    for ms in frame_ms:
        owed += ms
        n = 0
        while owed >= fixed_ms:
            owed -= fixed_ms
            n += 1
        steps.append(n)
    return steps`,
        hints: ['Keep one running total of the time owed, outside the loop over frames.', 'After adding a frame, take away fixed_ms as many times as it fits, counting each time.'],
        explain: `Carrying the leftover time is what keeps the simulation in step with the clock: over many frames the number of fixed steps equals the total time divided by the step length, even when no single frame is a whole number of steps. Integer milliseconds avoid floating-point rounding at the boundaries; real engines use seconds and a small tolerance.`,
        tests: [
          { call: 'physics_steps([16, 16, 16, 50])', expect: '[0, 1, 1, 2]' },
          { call: 'physics_steps([25, 25], 10)', expect: '[2, 3]' },
          { name: 'no frames', call: 'physics_steps([])', expect: '[]', hidden: true },
          { name: 'short frames add up', call: 'physics_steps([5] * 8)', expect: '[0, 0, 0, 1, 0, 0, 0, 1]', hidden: true },
          { name: 'a long frame', call: 'physics_steps([100])', expect: '[5]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Game Loop (Game Programming Patterns, Robert Nystrom)', url: 'https://gameprogrammingpatterns.com/game-loop.html', note: 'The loop, variable and fixed time steps, with diagrams' },
      { title: 'Fix Your Timestep (Glenn Fiedler)', url: 'https://gafferongames.com/post/fix_your_timestep/', note: 'The accumulator behind fixed physics steps' },
      { title: 'Time.deltaTime (Unity scripting reference)', url: 'https://docs.unity3d.com/ScriptReference/Time-deltaTime.html' },
      { title: 'MonoBehaviour.FixedUpdate (Unity scripting reference)', url: 'https://docs.unity3d.com/ScriptReference/MonoBehaviour.FixedUpdate.html' }
    ]
  },
  // =====================================================================================
  {
    id: 'vectors',
    branch: 'math',
    title: 'Vectors for Movement',
    short: 'Vectors',
    level: 1,
    minutes: 30,
    prereqs: ['game-loop'],
    tags: ['vector', 'vector3', 'direction', 'length', 'magnitude', 'normalize', 'dot product', 'move towards', 'unity'],
    summary: 'A vector is an arrow with a direction and a length. Games use vectors for positions, velocities and directions; normalizing, measuring length and the dot product answer most everyday movement questions.',
    why: `Almost every line of gameplay code asks a question about space: where is the enemy, which way should the player face, how far away is the goal, is the target in front of me? Vectors answer all of them with a handful of operations, so learning them once pays off in every script you write.`,
    objectives: [
      'Find the vector from one point to another, its length and its direction',
      'Move an object towards a target at a fixed speed without overshooting',
      'Use the sign of the dot product to tell whether something is in front or behind'
    ],
    sections: [
      { type: 'text', title: 'Points, arrows and lengths', md: `A **vector** is a list of numbers, such as (3, 4) in 2-D or (3, 4, 0) in 3-D. The same pair can mean a **point** (a position) or an **arrow** (a displacement: how to get from one point to another). Unity uses \`Vector2\` and \`Vector3\` for both.

The arrow from point A to point B is B − A, computed coordinate by coordinate. Its **length** (Unity: \`magnitude\`) is the distance between the points:

$$ ‖v‖ = √(v_x² + v_y²) $$
:::where
v — a vector with coordinates (v_x, v_y)
v_x — the vector's x coordinate (v_y: its y coordinate)
‖v‖ — the length of v: the straight-line distance it covers
:::

Dividing a vector by its length gives a vector of length 1 that points the same way, called a **direction** or a **normalized** vector (Unity: \`normalized\`). Multiply a direction by a speed and by Δt, and you get this frame's step.` },
      { type: 'code', title: 'Chasing an enemy', md: `The player at (2, 1) moves towards an enemy at (5, 5) at 4 units per second, for half a second.`, code: `import math

player = (2.0, 1.0)
enemy = (5.0, 5.0)

to_enemy = (enemy[0] - player[0], enemy[1] - player[1])        # the arrow from player to enemy
distance = math.sqrt(to_enemy[0] ** 2 + to_enemy[1] ** 2)       # its length
direction = (to_enemy[0] / distance, to_enemy[1] / distance)    # the same arrow, length 1

print("to enemy:", to_enemy)
print("distance:", distance)
print("direction:", direction, "with length", math.hypot(*direction))

speed, dt = 4.0, 0.5             # units per second, seconds
player = (player[0] + direction[0] * speed * dt, player[1] + direction[1] * speed * dt)
print("player after 0.5 s:", player)`, after: `The arrow (3, 4) has length 5, so the direction is (0.6, 0.8). Half a second at 4 units per second is a step of 2 units along it: (1.2, 1.6).`, predict: { type: 'mcq', q: 'What distance between the player and the enemy does the code print?', options: ['5.0', '7.0', '25.0', '3.5'], answer: 0, explain: 'The arrow is (3, 4), and √(3² + 4²) = √25 = 5. Adding the coordinates (7) or skipping the square root (25) are the usual slips.', expect: ['distance: 5.0'] } },
      { type: 'text', title: 'In front or behind: the dot product', md: `The **dot product** multiplies two vectors coordinate by coordinate and adds the results:

$$ a · b = a_x b_x + a_y b_y $$
:::where
a, b — two vectors
a_x, a_y — the coordinates of a (b_x, b_y: those of b)
a · b — their dot product: a single number
:::

For two directions (length 1), the dot product is the cosine of the angle between them: 1 when they point the same way, 0 when they are at right angles, −1 when they point opposite ways. So a guard facing direction f sees a target in direction d in front when f · d > 0 and behind when it is negative. Comparing with a threshold such as 0.7 instead of 0 narrows the field of view (about 45 degrees to each side).` },
      { type: 'code', title: 'Ahead, beside, behind', code: `facing = (1.0, 0.0)              # the guard looks along +x
for name, d in [("ahead", (0.6, 0.8)), ("beside", (0.0, 1.0)), ("behind", (-0.6, -0.8))]:
    dot = facing[0] * d[0] + facing[1] * d[1]
    print(f"{name:<6} dot = {dot:+.1f}")`, after: `Positive means in front, zero means exactly beside, negative means behind.` },
      { type: 'code', title: 'The same in Unity', lang: 'csharp', run: false, md: `Unity's \`Vector3\` has all of these built in: \`magnitude\`, \`normalized\`, \`Vector3.Dot\` and \`Vector3.MoveTowards\`, which steps towards a target without overshooting it.`, code: `using UnityEngine;

public class Chaser : MonoBehaviour
{
    [SerializeField] Transform target;
    [SerializeField] float speed = 4f;     // units per second

    void Update()
    {
        Vector3 toTarget = target.position - transform.position;
        bool inFront = Vector3.Dot(transform.forward, toTarget.normalized) > 0f;
        if (inFront)
            transform.position = Vector3.MoveTowards(transform.position, target.position, speed * Time.deltaTime);
    }
}` },
      { type: 'check', question: { type: 'mcq', q: 'A guard faces (0, 1). A target lies in direction (0, −1) from the guard. What is the dot product of the two directions?', options: ['−1', '0', '1', '2'], answer: 0, explain: '0 · 0 + 1 · (−1) = −1: the directions are opposite, so the target is directly behind.' } }
    ],
    when: `- Positions, velocities, directions, forces and offsets are all vectors.
- Use the length for distances and ranges, the normalized vector for directions, and the dot product for "in front or behind" and field-of-view checks.`,
    pitfalls: `- Normalizing a zero-length vector (two objects at the same spot) divides by zero; check the distance first.
- Moving by direction · speed · Δt past the target makes objects jitter around it; stop at the target, as MoveTowards does.
- Comparing distances with the square root every frame: comparing squared lengths gives the same answer more cheaply.`,
    takeaways: [
      'The arrow from A to B is B − A; its length is the distance between the points.',
      'Dividing a vector by its length gives a direction of length 1.',
      'The sign of the dot product of two directions says whether they point roughly the same way (positive) or apart (negative).'
    ],
    quiz: [
      { type: 'fill', q: 'What is the length of the vector (6, 8)?', answer: '10', explain: '√(6² + 8²) = √(36 + 64) = √100 = 10.' },
      { type: 'mcq', q: 'The player is at (1, 2) and a coin at (4, 6). Which vector points from the player to the coin?', options: ['(3, 4)', '(−3, −4)', '(5, 8)', '(4, 6)'], answer: 0, explain: 'Coin minus player: (4 − 1, 6 − 2) = (3, 4). The reverse order gives the arrow back from the coin.' },
      { type: 'mcq', q: 'What does normalizing a vector do?', options: ['Keeps its direction and sets its length to 1', 'Sets every coordinate to 1', 'Rotates it by 90 degrees', 'Makes all its coordinates positive'], answer: 0, explain: 'Dividing by the length keeps the direction and scales the length to 1, ready to be multiplied by a speed.' },
      { type: 'tf', q: 'If the dot product of a guard\'s facing direction and the direction to a target is negative, the target is behind the guard.', answer: true, explain: 'A negative dot product means the angle between the directions is more than 90 degrees.' },
      { type: 'mcq', q: 'Why does MoveTowards take a maximum step instead of always moving the full step?', options: ['So the object stops on the target instead of overshooting it', 'To save memory', 'Because vectors cannot be longer than 1', 'To make the movement random'], answer: 0, explain: 'When the target is closer than one step, the object lands exactly on it instead of passing it and turning back every frame.' }
    ],
    problems: [
      {
        id: 'move-towards',
        title: 'Move Towards a Target',
        difficulty: 'easy',
        prompt: `Write \`move_towards(pos, target, max_step)\`, like Unity's \`Vector3.MoveTowards\` in 2-D. Points are \`(x, y)\` tuples. Move from \`pos\` straight towards \`target\` by \`max_step\` units, but never past it: if the target is at most \`max_step\` away, return the target itself. Return a tuple of floats.

Example: \`move_towards((0, 0), (3, 4), 1)\` returns \`(0.6, 0.8)\`: the target is 5 units away in direction (0.6, 0.8), and the step is 1 unit long.`,
        starter: `import math

def move_towards(pos, target, max_step):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    dist = math.hypot(dx, dy)
    return (pos[0] + dx / dist * max_step, pos[1] + dy / dist * max_step)`,
        solution: `import math

def move_towards(pos, target, max_step):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    dist = math.hypot(dx, dy)
    if dist <= max_step:
        return (float(target[0]), float(target[1]))
    return (pos[0] + dx / dist * max_step, pos[1] + dy / dist * max_step)`,
        hints: ['Compute the distance to the target first.', 'If the distance is not more than max_step, return the target: that also avoids dividing by zero when pos equals target.'],
        explain: `Checking the distance first handles both edge cases at once: the step that would overshoot, and the zero-length vector that cannot be normalized. Called every frame with max_step = speed · Δt, this gives smooth movement that stops exactly on the target.`,
        tests: [
          { call: 'move_towards((0, 0), (3, 4), 1)', expect: '(0.6, 0.8)' },
          { call: 'move_towards((0, 0), (3, 4), 10)', expect: '(3.0, 4.0)' },
          { name: 'already there', call: 'move_towards((1, 1), (1, 1), 2)', expect: '(1.0, 1.0)', hidden: true },
          { name: 'a longer move', call: 'move_towards((2, -1), (-4, 7), 5)', expect: '(-1.0, 3.0)', hidden: true },
          { name: 'exactly one step away', call: 'move_towards((0, 0), (3, 4), 5)', expect: '(3.0, 4.0)', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Vector3 (Unity scripting reference)', url: 'https://docs.unity3d.com/ScriptReference/Vector3.html' },
      { title: 'Vector3.MoveTowards (Unity scripting reference)', url: 'https://docs.unity3d.com/ScriptReference/Vector3.MoveTowards.html' },
      { title: 'Vector3.Dot (Unity scripting reference)', url: 'https://docs.unity3d.com/ScriptReference/Vector3.Dot.html' }
    ]
  },
  // =====================================================================================
  {
    id: 'collisions',
    branch: 'play',
    title: 'Collision Detection Basics',
    short: 'Collisions',
    level: 2,
    minutes: 35,
    prereqs: ['vectors'],
    tags: ['collision', 'collider', 'aabb', 'bounding box', 'circle', 'overlap', 'trigger', 'tunneling', 'continuous collision detection', 'unity'],
    summary: 'Games test collisions with simple shapes: circles, spheres and axis-aligned boxes. Two cheap tests cover most cases, and fast objects need extra care so they do not pass straight through thin walls.',
    why: `Picking up a coin, landing on a platform, being hit by a bullet: most gameplay starts with the question "do these two things touch?" Engines answer it thousands of times per frame, so the tests must be cheap. Knowing how they work explains the colliders you set up in Unity and the strange bugs, such as bullets passing through walls, that appear when they are misused.`,
    objectives: [
      'Test two circles (or spheres) for overlap without a square root',
      'Test two axis-aligned boxes for overlap, axis by axis',
      'Explain tunneling and how continuous collision detection prevents it'
    ],
    sections: [
      { type: 'text', title: 'Simple shapes stand in for complex ones', md: `A character model may have thousands of triangles, but its **collider** is usually a capsule, a sphere or a box. Testing simple shapes is fast, and players rarely notice the difference. Two tests cover most cases.

**Circles (and spheres)** overlap when the distance between their centres is at most the sum of their radii. Comparing squared values avoids the square root:

$$ (x₁ − x₂)² + (y₁ − y₂)² ≤ (r₁ + r₂)² $$
:::where
x₁, y₁ — the centre of the first circle (x₂, y₂: of the second)
r₁, r₂ — the two radii
:::

**Axis-aligned boxes** (AABBs: boxes whose sides are parallel to the axes) overlap when their extents overlap on *every* axis. On the x axis, box a spans from its minimum to its maximum x, and the two spans overlap when each box starts before the other ends. A gap on any single axis means no collision.` },
      { type: 'code', title: 'Two tests', code: `def circles_touch(c1, r1, c2, r2):
    dx, dy = c1[0] - c2[0], c1[1] - c2[1]
    return dx * dx + dy * dy <= (r1 + r2) ** 2      # no square root needed

def boxes_overlap(a, b):
    # a box is (min_x, min_y, max_x, max_y)
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]

print("circles 1.5 apart, radii 1.0 and 0.6:", circles_touch((0, 0), 1.0, (1.5, 0), 0.6))
print("circles 5 apart, radii 1.0 and 2.0:  ", circles_touch((0, 0), 1.0, (3, 4), 2.0))
print("boxes sharing a corner region:       ", boxes_overlap((0, 0, 2, 2), (1, 1, 3, 3)))
print("boxes with a gap on the x axis:      ", boxes_overlap((0, 0, 2, 2), (2.5, 0, 4, 2)))`, after: `The first circles are 1.5 apart with radii adding up to 1.6, so they touch. The last boxes overlap on y but leave a gap from 2 to 2.5 on x, which is enough to rule out a collision.` },
      { type: 'text', title: 'Tunneling: when fast objects skip a wall', md: `Collision tests look at positions **at the end of each step**. A bullet that moves 30 units per step can be in front of a 1-unit wall at one step and behind it at the next, without ever being inside it. This is called **tunneling**.

The usual fixes:

- make thin walls thicker, or slow the object down;
- test the whole path swept during the step, not just its end: **continuous collision detection**. In Unity, set the Rigidbody's *Collision Detection* to *Continuous* for fast objects such as bullets;
- for instant hits, cast a ray along the path (\`Physics.Raycast\`) instead of moving a physical bullet.` },
      { type: 'code', title: 'A bullet passes through a wall', code: `x, speed = 0.0, 30.0             # units per step
wall = (50.0, 51.0)               # the wall spans x from 50 to 51
for frame in range(1, 5):
    x += speed
    inside = wall[0] <= x <= wall[1]
    print(f"frame {frame}: x = {x:5.1f}, inside the wall: {inside}")`, after: `The bullet jumps from 30 to 60. It was never inside the wall at the end of a step, so a test of end positions never sees the hit.`, predict: { type: 'mcq', q: 'The bullet moves 30 units per step towards a wall from x = 50 to x = 51. At which step does the code report it inside the wall?', options: ['At none of the four steps', 'At step 2', 'At step 1', 'At every step after 2'], answer: 0, explain: 'The end positions are 30, 60, 90 and 120. None lies between 50 and 51, so the hit is missed: that is tunneling.', expect: ['frame 2: x =  60.0, inside the wall: False'] } },
      { type: 'code', title: 'Collisions in Unity', lang: 'csharp', run: false, md: `In Unity you add collider components and react to events. A collider marked *Is Trigger* reports overlaps without blocking movement, which suits pickups. At least one of the two objects needs a Rigidbody for the events to fire.`, code: `using UnityEngine;

// On the coin: a collider with "Is Trigger" ticked.
// On the player: a collider and a Rigidbody, and the tag "Player".
public class Coin : MonoBehaviour
{
    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
            Destroy(gameObject);       // collect the coin
    }
}` },
      { type: 'check', question: { type: 'mcq', q: 'Box a spans x from 0 to 2 and y from 0 to 2. Box b spans x from 1 to 3 and y from 5 to 6. Do they overlap?', options: ['No: they leave a gap on the y axis', 'Yes: they overlap on the x axis', 'Yes: they are the same size', 'It cannot be told without their areas'], answer: 0, explain: 'Overlap needs every axis. Their x spans overlap (1 to 2), but on y one ends at 2 and the other starts at 5.' } }
    ],
    when: `- Use spheres or capsules for characters, boxes for crates and walls, and mesh colliders only for complex static scenery.
- Use triggers for pickups, zones and checkpoints; solid colliders for things that block movement.
- Switch fast-moving objects to continuous collision detection, or use raycasts for instant hits.`,
    pitfalls: `- Forgetting the Rigidbody: two colliders without one send no collision or trigger events.
- Testing only one axis of two boxes: a gap on any axis rules out a collision.
- Moving physics objects by setting transform.position each frame: that bypasses the physics engine; move them with the Rigidbody instead.`,
    takeaways: [
      'Two circles touch when the squared distance between their centres is at most the squared sum of their radii.',
      'Two axis-aligned boxes overlap only if their spans overlap on every axis.',
      'Fast objects can tunnel through thin colliders between steps; continuous detection or raycasts prevent it.'
    ],
    quiz: [
      { type: 'mcq', q: 'Two circles have radii 2 and 3, and their centres are 4 units apart. Do they overlap?', options: ['Yes: 4 is less than 2 + 3', 'No: 4 is more than 3', 'No: 4 is more than 2', 'Only if they move'], answer: 0, explain: 'They overlap when the distance is at most the sum of the radii: 4 ≤ 5.' },
      { type: 'mcq', q: 'Why do collision tests often compare squared distances?', options: ['It gives the same answer without computing a square root', 'Squared distances are more accurate', 'Unity cannot compute square roots', 'It makes circles behave like boxes'], answer: 0, explain: 'Both sides are non-negative, so comparing their squares preserves the result and skips the costly square root.' },
      { type: 'tf', q: 'Two axis-aligned boxes whose x spans overlap must be colliding.', answer: false, explain: 'They must also overlap on y (and on z in 3-D). A gap on any one axis means no collision.' },
      { type: 'mcq', q: 'A fast bullet sometimes passes straight through a thin wall. What is this called?', options: ['Tunneling', 'Aliasing', 'Clipping planes', 'Z-fighting'], answer: 0, explain: 'Between two steps the bullet jumps from one side of the wall to the other, so no step finds it inside.' },
      { type: 'mcq', q: 'In Unity, which collider setting reports overlaps without blocking movement?', options: ['Is Trigger', 'Is Kinematic', 'Static batching', 'Continuous detection'], answer: 0, explain: 'A trigger collider sends OnTriggerEnter and similar events but lets objects pass through, which suits pickups and zones.' }
    ],
    problems: [
      {
        id: 'overlapping-pairs',
        title: 'Find Overlapping Boxes',
        difficulty: 'medium',
        prompt: `Boxes are given as \`(min_x, min_y, max_x, max_y)\`. Two boxes overlap when their spans overlap on both axes; touching edges count as overlapping.

Write \`overlapping_pairs(boxes)\` that returns every pair of indices \`(i, j)\` with \`i < j\` whose boxes overlap, in increasing order of i, then j.

Example: for \`[(0, 0, 2, 2), (1, 1, 3, 3), (5, 0, 6, 1), (1, 5, 2, 6)]\` the answer is \`[(0, 1)]\`. Boxes 0 and 3 overlap on x but not on y.`,
        starter: `def overlapping_pairs(boxes):
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[0] <= b[2] and b[0] <= a[2]:       # only the x axis so far
                pairs.append((i, j))
    return pairs`,
        solution: `def overlapping_pairs(boxes):
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                pairs.append((i, j))
    return pairs`,
        hints: ['Spans [a0, a1] and [b0, b1] overlap when a0 <= b1 and b0 <= a1.', 'Check the same condition on the y axis too.'],
        explain: `Checking every pair takes about n² / 2 tests, fine for dozens of objects. Engines handle thousands by first sorting or bucketing the boxes (a broad phase, such as sweep and prune or a grid) so that only nearby pairs get the exact test.`,
        tests: [
          { call: 'overlapping_pairs([(0, 0, 2, 2), (1, 1, 3, 3), (5, 0, 6, 1), (1, 5, 2, 6)])', expect: '[(0, 1)]' },
          { call: 'overlapping_pairs([(0, 0, 1, 1), (1, 0, 2, 1)])', expect: '[(0, 1)]' },
          { name: 'no boxes', call: 'overlapping_pairs([])', expect: '[]', hidden: true },
          { name: 'three boxes in one spot', call: 'overlapping_pairs([(0, 0, 1, 1)] * 3)', expect: '[(0, 1), (0, 2), (1, 2)]', hidden: true },
          { name: 'a box inside another', call: 'overlapping_pairs([(0, 0, 10, 10), (4, 4, 5, 5)])', expect: '[(0, 1)]', hidden: true },
          { name: 'apart on y only', call: 'overlapping_pairs([(0, 0, 4, 1), (1, 3, 2, 4)])', expect: '[]', hidden: true }
        ]
      }
    ],
    refs: [
      { title: 'Bounds.Intersects (Unity scripting reference)', url: 'https://docs.unity3d.com/ScriptReference/Bounds.Intersects.html' },
      { title: 'Collider.OnTriggerEnter (Unity scripting reference)', url: 'https://docs.unity3d.com/ScriptReference/Collider.OnTriggerEnter.html' },
      { title: 'Rigidbody.collisionDetectionMode (Unity scripting reference)', url: 'https://docs.unity3d.com/ScriptReference/Rigidbody-collisionDetectionMode.html', note: 'Continuous collision detection for fast objects' }
    ]
  }
  ]);
})(typeof globalThis !== 'undefined' ? globalThis : window);
