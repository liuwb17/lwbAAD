"""Measured layout constraints, not a rectangle construction or search algorithm."""
from .problem import W,judge,parse_input


def layout_diagnostics(input_text,output_text,limit=5):
    if judge(input_text,output_text)['status']!='AC':
        raise ValueError('Geometry diagnostics require a legal layout')
    points=parse_input(input_text)
    values=list(map(int,output_text.split()))
    rects=[tuple(values[i:i+4]) for i in range(0,len(values),4)]
    areas=[(c-a)*(d-b) for a,b,c,d in rects]
    contained=[a<=x<c and b<=y<d for (x,y,_),(a,b,c,d) in zip(points,rects)]
    scores=[1-(1-min(area,r)/max(area,r))**2 if hit else 0.
            for area,(_,_,r),hit in zip(areas,points,contained)]
    worst=[]
    for i in sorted(range(len(points)),key=lambda k:scores[k])[:limit]:
        a,b,c,d=rects[i]
        candidates={'x_min':[(a,None)],'x_max':[(W-c,None)],
                    'y_min':[(b,None)],'y_max':[(W-d,None)]}
        for j,(e,f,g,h) in enumerate(rects):
            if i==j:continue
            if max(b,f)<min(d,h):
                if g<=a:candidates['x_min'].append((a-g,j))
                if e>=c:candidates['x_max'].append((e-c,j))
            if max(a,e)<min(c,g):
                if h<=b:candidates['y_min'].append((b-h,j))
                if f>=d:candidates['y_max'].append((f-d,j))
        edges={}
        for edge,items in candidates.items():
            distance=min(gap for gap,_ in items)
            hits=[j for gap,j in items if gap==distance]
            blockers=[j for j in hits if j is not None]
            edges[edge]={'free_expansion_distance':distance,'board_is_limit':None in hits,
                         'blocker_count':len(blockers),'nearest_rectangles':[
                             {'index':j,'area':areas[j],'target':points[j][2],
                              'satisfaction':round(scores[j],9)} for j in blockers[:3]]}
        worst.append({'index':i,'rectangle':rects[i],'area':areas[i],'target':points[i][2],
                      'contains_anchor':contained[i],'satisfaction':round(scores[i],9),
                      'score_loss_contribution':round(1e9*(1-scores[i])/len(points),3),
                      'fixed_layout_edge_constraints':edges})
    return {'indexing':'zero-based input order','unused_board_area':W*W-sum(areas),
            'measurement':'Maximum single-edge outward distance with every other edge and rectangle fixed. These are measured constraints, not proposed moves. At most three tied blockers are listed per edge.',
            'worst_rectangles':worst}
