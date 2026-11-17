"""Optional rectangular founder zones and exact boundary-biased food placement."""


def zone(position, bounds):
    x, y = position
    left, top, right, bottom = bounds
    if left < x < right and top < y < bottom:
        return 'inside'
    if x < left or x > right or y < top or y > bottom:
        return 'outside'
    return 'boundary'


def resource_pools(free, placement):
    bounds = placement['enclosure_bounds']
    left, top, right, bottom = bounds
    width = placement['resources']['boundary_width']
    pools = {'inside': [], 'outside': [], 'far': []}
    for p in free:
        side = zone(p, bounds)
        if side == 'boundary':
            continue
        x, y = p
        distance = (min(x-left, right-x, y-top, bottom-y) if side == 'inside'
                    else max(left-x, x-right, top-y, y-bottom))
        pools[side if distance <= width else 'far'].append(p)
    return pools


def resource_counts(count, rules):
    near = int(count * rules['boundary_fraction'] + .5)
    inside = int(near * rules['inside_fraction'] + .5)
    return {'inside': inside, 'outside': near-inside, 'far': count-near}


def validate_placement(config, unavailable):
    from .config import obj, num, fail
    p = config['placement']
    obj(p, 'placement', ('enclosure_bounds', 'species', 'resources'))
    if config['obstacles']['generation'] != 'explicit':
        fail('placement', 'requires explicit obstacles for predictable zone capacity')
    bounds = p['enclosure_bounds']
    if not isinstance(bounds, list) or len(bounds) != 4:
        fail('placement.enclosure_bounds', 'expected [left, top, right, bottom]')
    w = config['world']
    for i, n in enumerate(bounds):
        num(n, f'placement.enclosure_bounds[{i}]', 0, w['width' if i%2 == 0 else 'height']-1, True)
    if bounds[2]-bounds[0] < 2 or bounds[3]-bounds[1] < 2:
        fail('placement.enclosure_bounds', 'must enclose at least one interior cell')
    obj(p['species'], 'placement.species', tuple(config['species']))
    for name, side in p['species'].items():
        if side not in ('inside', 'outside'):
            fail('placement.species.'+name, 'expected inside or outside')
    r = p['resources']
    obj(r, 'placement.resources', ('boundary_fraction', 'boundary_width', 'inside_fraction'))
    num(r['boundary_fraction'], 'placement.resources.boundary_fraction', 0, 1)
    num(r['inside_fraction'], 'placement.resources.inside_fraction', 0, 1)
    num(r['boundary_width'], 'placement.resources.boundary_width', 1, 32, True)
    free = [(x,y) for y in range(w['height']) for x in range(w['width']) if (x,y) not in unavailable]
    for side in ('inside', 'outside'):
        demand = sum(s['initial_population'] for name,s in config['species'].items() if p['species'][name] == side)
        if demand > sum(zone(cell,bounds) == side for cell in free):
            fail('placement.species', f'insufficient {side} dry cells for founders')
    pools = resource_pools(free, p)
    for side,count in resource_counts(config['resources']['patch_count'], r).items():
        if count > len(pools[side]):
            fail('placement.resources', f'insufficient {side} dry cells for requested food allocation')
