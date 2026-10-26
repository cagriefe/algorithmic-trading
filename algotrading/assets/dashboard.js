(function () {
  'use strict';

  document.querySelectorAll('table.sortable').forEach(function (table) {
    var headers = table.tHead.rows[0].cells;
    Array.prototype.forEach.call(headers, function (header, column) {
      header.addEventListener('click', function () {
        var ascending = header.dataset.dir !== 'asc';
        Array.prototype.forEach.call(headers, function (h) { delete h.dataset.dir; });
        header.dataset.dir = ascending ? 'asc' : 'desc';

        var body = table.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = a.cells[column].dataset.value;
          var y = b.cells[column].dataset.value;
          if (x === '' || y === '') {
            return x === y ? 0 : x === '' ? 1 : -1;  // missing values always last
          }
          var nx = Number(x);
          var ny = Number(y);
          var order = !isNaN(nx) && !isNaN(ny) ? nx - ny : String(x).localeCompare(String(y));
          return ascending ? order : -order;
        });
        rows.forEach(function (row) { body.appendChild(row); });
      });
    });
  });
})();
