"""Filter the full visible library before pagination."""
def page_workspace_items(items, *, q='', source='', file_type='', bucket='', modified_after='', sort='created_desc', page=1, page_size=50):
    sources = sorted({str(i['source']) for i in items})
    query = q.strip().lower()
    selected = [i for i in items if
        (not query or query in i['display_name'].lower() or query in i['storage_path'].lower())
        and (not source or source.lower() == str(i['source']).lower())
        and (not file_type or file_type.lower() == str(i['category']).lower())
        and (not bucket or bucket == i['bucket'])
        and (not modified_after or i['created_at'] >= modified_after)]
    if sort == 'name_asc':
        selected.sort(key=lambda i: (i['display_name'].lower(), i['storage_path']))
    elif sort == 'size_desc':
        selected.sort(key=lambda i: (i['size'], i['storage_path']), reverse=True)
    else:
        selected.sort(key=lambda i: (i['created_at'], i['storage_path']), reverse=True)
    total = len(selected)
    page = min(max(1, page), max(1, (total + page_size - 1)//page_size))
    start = (page-1)*page_size
    return {'items': selected[start:start+page_size], 'total': total,
            'total_size': sum(i['size'] for i in selected), 'page': page,
            'page_size': page_size, 'sources': sources}
