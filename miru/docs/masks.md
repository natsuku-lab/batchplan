# A mask model's masks

A look tells things apart by geometry first and needs no model. Depth cannot join a thing's
parts that show no connection, as a rack's slabs, and on a real camera's noise it loses some
splits. A mask model's instance masks do both.

`View::masks` (and `Camera::masks` for `observe`) takes one number per pixel, in the depth's
order: the thing the model says the pixel shows, and 0 where it says none. A number means a thing
in that view only. Leave it empty for a view without a model. miru runs no model and links none:
the numbers are data from whatever the caller runs.

What the masks change, within one view:

- **Split.** Readings that depth makes one surface and the masks two things are cut apart, unless
  they are a mask's fringe, a sliver too thin to be a thing.
- **Join.** What the view shows mostly in one mask is one object.
- **Apart.** What it shows mostly in two masks is two objects, whatever another view shows.

The robot's readings and those outside the look's corners stay what they are, and a known shape's
readings go to a mask's thing only where they lie inside it. With every mask empty or 0, a look is
geometry's alone.

A model that needs prompts, points or boxes takes two looks of one moment: look without masks,
take a prompt from each object's pixels (its `Label::Object` labels), run the model, and look
again with its masks after the same scene before.

Masks need not be exact to the pixel: [status](status.md#masks-off-by-a-few-pixels) scores masks
moved 1 and 3 pixels in and out.
